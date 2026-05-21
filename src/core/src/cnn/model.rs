use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::Config;
use burn::tensor::activation::{relu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};

use crate::types::ALPHABET_LEN;

pub const CNN_1D_FEATURES: usize = 16 * 60;
pub const CNN_2D_FEATURES: usize = 32 * 24 * 24;
pub const CNN_FUSED_FEATURES: usize = CNN_1D_FEATURES + CNN_2D_FEATURES;

#[derive(Config, Debug)]
pub struct PicoDualCnnConfig {
    #[config(default = 128)]
    pub hidden_size: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CnnModelMetadata {
    pub architecture: String,
    pub labels: Vec<String>,
    pub hidden_size: usize,
    pub one_d_shape: [usize; 2],
    pub two_d_shape: [usize; 3],
    pub trained_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub train_samples: usize,
    pub validation_samples: usize,
}

impl CnnModelMetadata {
    pub fn new(config: &PicoDualCnnConfig) -> Self {
        Self {
            architecture: "pico-dual-cnn-v1".to_string(),
            labels: (b'A'..=b'Z')
                .map(|label| (label as char).to_string())
                .collect(),
            hidden_size: config.hidden_size,
            one_d_shape: [3, 64],
            two_d_shape: [1, 28, 28],
            trained_epochs: 0,
            batch_size: 0,
            learning_rate: 0.0,
            seed: 0,
            train_samples: 0,
            validation_samples: 0,
        }
    }

    pub fn config(&self) -> PicoDualCnnConfig {
        PicoDualCnnConfig::new().with_hidden_size(self.hidden_size)
    }
}

#[derive(Module, Debug)]
pub struct PicoDualCnn<B: Backend> {
    one_d_conv1: Conv1d<B>,
    one_d_conv2: Conv1d<B>,
    two_d_conv1: Conv2d<B>,
    two_d_conv2: Conv2d<B>,
    classifier_hidden: Linear<B>,
    classifier_out: Linear<B>,
}

impl PicoDualCnnConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PicoDualCnn<B> {
        PicoDualCnn {
            one_d_conv1: Conv1dConfig::new(3, 8, 3).init(device),
            one_d_conv2: Conv1dConfig::new(8, 16, 3).init(device),
            two_d_conv1: Conv2dConfig::new([1, 16], [3, 3]).init(device),
            two_d_conv2: Conv2dConfig::new([16, 32], [3, 3]).init(device),
            classifier_hidden: LinearConfig::new(CNN_FUSED_FEATURES, self.hidden_size).init(device),
            classifier_out: LinearConfig::new(self.hidden_size, ALPHABET_LEN).init(device),
        }
    }
}

impl<B: Backend> PicoDualCnn<B> {
    pub fn forward(&self, one_d: Tensor<B, 3>, two_d: Tensor<B, 4>) -> Tensor<B, 2> {
        let one_d = relu(self.one_d_conv1.forward(one_d));
        let one_d = relu(self.one_d_conv2.forward(one_d));
        let one_d = one_d.flatten::<2>(1, 2);

        let two_d = relu(self.two_d_conv1.forward(two_d));
        let two_d = relu(self.two_d_conv2.forward(two_d));
        let two_d = two_d.flatten::<2>(1, 3);

        let fused = Tensor::cat(vec![one_d, two_d], 1);
        let hidden = relu(self.classifier_hidden.forward(fused));
        self.classifier_out.forward(hidden)
    }

    pub fn predict_probabilities(&self, one_d: Tensor<B, 3>, two_d: Tensor<B, 4>) -> Tensor<B, 2> {
        softmax(self.forward(one_d, two_d), 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn forward_returns_alphabet_logits() {
        let device = Default::default();
        let model = PicoDualCnnConfig::new().init::<TestBackend>(&device);
        let one_d = Tensor::<TestBackend, 3>::zeros([2, 3, 64], &device);
        let two_d = Tensor::<TestBackend, 4>::zeros([2, 1, 28, 28], &device);
        let output = model.forward(one_d, two_d);
        assert_eq!(output.dims(), [2, 26]);
    }

    #[test]
    fn inference_softmax_sums_to_one() {
        let device = Default::default();
        let model = PicoDualCnnConfig::new().init::<TestBackend>(&device);
        let one_d = Tensor::<TestBackend, 3>::zeros([1, 3, 64], &device);
        let two_d = Tensor::<TestBackend, 4>::zeros([1, 1, 28, 28], &device);
        let probabilities = model.predict_probabilities(one_d, two_d);
        let values = probabilities.into_data().to_vec::<f32>().unwrap();
        let sum = values.iter().sum::<f32>();
        assert!((sum - 1.0).abs() < 1e-4);
    }
}
