use std::fs;
use std::path::{Path, PathBuf};

use burn::backend::{Autodiff, NdArray};
#[cfg(feature = "gpu")]
use burn::backend::Wgpu;
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn_store::{ModuleSnapshot, SafetensorsStore};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::cnn::model::{CnnModelMetadata, PicoDualCnn, PicoDualCnnConfig};
use crate::data::bridge::batcher::{batch_dual, CnnBatch, DualSample};
use crate::data::pipeline::preprocess_file;
use crate::logger::log_progress;
use crate::types::ALPHABET_LEN;

#[derive(Clone, Debug)]
pub struct TrainCnnOptions {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub use_cpu: bool,
}

impl Default for TrainCnnOptions {
    fn default() -> Self {
        Self {
            epochs: 20,
            batch_size: 32,
            learning_rate: 0.001,
            seed: 42,
            use_cpu: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BalancedSplit {
    pub train: Vec<DualSample>,
    pub validation: Vec<DualSample>,
    pub per_label_count: usize,
}

pub fn balanced_split(samples: Vec<DualSample>, seed: u64) -> Result<BalancedSplit, String> {
    let mut by_label = (0..ALPHABET_LEN).map(|_| Vec::new()).collect::<Vec<_>>();
    for sample in samples {
        if sample.label_index >= ALPHABET_LEN {
            return Err(format!("invalid label index {}", sample.label_index));
        }
        by_label[sample.label_index].push(sample);
    }

    let min_count = by_label
        .iter()
        .map(Vec::len)
        .min()
        .ok_or_else(|| "no labels available".to_string())?;
    if min_count < 2 {
        return Err(format!(
            "all 26 labels need at least 2 valid samples; smallest label has {min_count}"
        ));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut train = Vec::new();
    let mut validation = Vec::new();

    for label_samples in by_label.iter_mut() {
        label_samples.shuffle(&mut rng);
        label_samples.truncate(min_count);
        let train_count = ((min_count as f64) * 0.8).floor() as usize;
        let train_count = train_count.clamp(1, min_count - 1);
        for (index, sample) in label_samples.drain(..).enumerate() {
            if index < train_count {
                train.push(sample);
            } else {
                validation.push(sample);
            }
        }
    }

    train.shuffle(&mut rng);
    validation.shuffle(&mut rng);

    Ok(BalancedSplit {
        train,
        validation,
        per_label_count: min_count,
    })
}

fn train_batch<B: AutodiffBackend>(
    model: PicoDualCnn<B>,
    optimizer: &mut impl Optimizer<PicoDualCnn<B>, B>,
    batch: CnnBatch<B>,
    learning_rate: f64,
) -> PicoDualCnn<B> {
    let logits = model.forward(batch.one_d, batch.two_d);
    let loss = CrossEntropyLossConfig::new()
        .with_logits(true)
        .init(&logits.device())
        .forward(logits, batch.labels);
    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    optimizer.step(learning_rate, model, grads)
}

fn train_cnn_with_backend<B: AutodiffBackend>(
    split: BalancedSplit,
    model_out: &Path,
    config_out: &Path,
    options: TrainCnnOptions,
    device: &B::Device,
) -> Result<CnnModelMetadata, String> {
    let config = PicoDualCnnConfig::new();
    let mut model = config.init::<B>(device);
    let mut optimizer = AdamConfig::new().init();

    log_progress(
        "train-cnn",
        0,
        None,
        &format!("starting training for {} epochs", options.epochs),
    );

    let batch_size = options.batch_size.max(1);
    let num_batches = (split.train.len() + batch_size - 1) / batch_size;

    for epoch in 0..options.epochs {
        log_progress(
            "train-cnn",
            (epoch as f64 / options.epochs as f64 * 100.0) as u8,
            None,
            &format!("starting epoch {}/{}", epoch + 1, options.epochs),
        );

        for (batch_idx, chunk) in split.train.chunks(batch_size).enumerate() {
            let batch = batch_dual::<B>(chunk, device)?;
            model = train_batch(model, &mut optimizer, batch, options.learning_rate);

            // Log every batch for the first 5 batches of the first epoch, then every 10
            let log_freq = if epoch == 0 && batch_idx < 5 { 1 } else { 10 };
            if batch_idx == num_batches - 1 || (batch_idx + 1) % log_freq == 0 {
                let percent = ((epoch as f64 + (batch_idx as f64 + 1.0) / num_batches as f64)
                    / options.epochs as f64
                    * 100.0) as u8;
                log_progress(
                    "train-cnn",
                    percent,
                    None,
                    &format!(
                        "epoch {}: batch {}/{}",
                        epoch + 1,
                        batch_idx + 1,
                        num_batches
                    ),
                );
            }
        }
    }

    let mut metadata = CnnModelMetadata::new(&config);
    metadata.trained_epochs = options.epochs;
    metadata.batch_size = options.batch_size;
    metadata.learning_rate = options.learning_rate;
    metadata.seed = options.seed;
    metadata.train_samples = split.train.len();
    metadata.validation_samples = split.validation.len();

    let valid_model = model.valid();
    save_safetensors(&valid_model, model_out)?;
    let metadata_json = serde_json::to_string_pretty(&metadata)
        .map_err(|error| format!("failed to serialize model config: {error}"))?;
    fs::write(config_out, metadata_json)
        .map_err(|error| format!("failed to write {}: {error}", config_out.display()))?;
    Ok(metadata)
}

pub fn train_cnn_from_file(
    sources: &[PathBuf],
    model_out: &Path,
    config_out: &Path,
    options: TrainCnnOptions,
) -> Result<CnnModelMetadata, String> {
    let mut all_paired = Vec::new();

    for source in sources {
        let output = preprocess_file(source)?;
        let paired =
            crate::data::bridge::batcher::pair_samples(&output.one_d_cnn, &output.two_d_cnn)?;
        all_paired.extend(paired);
    }

    let split = balanced_split(all_paired, options.seed)?;

    if options.use_cpu {
        train_cnn_with_backend::<Autodiff<NdArray<f32>>>(
            split,
            model_out,
            config_out,
            options,
            &Default::default(),
        )
    } else {
        train_cnn_with_gpu_backend(split, model_out, config_out, options)
    }
}

#[cfg(feature = "gpu")]
fn train_cnn_with_gpu_backend(
    split: BalancedSplit,
    model_out: &Path,
    config_out: &Path,
    options: TrainCnnOptions,
) -> Result<CnnModelMetadata, String> {
    train_cnn_with_backend::<Autodiff<Wgpu<f32, i32>>>(
        split,
        model_out,
        config_out,
        options,
        &burn::backend::wgpu::WgpuDevice::default(),
    )
}

#[cfg(not(feature = "gpu"))]
fn train_cnn_with_gpu_backend(
    _split: BalancedSplit,
    _model_out: &Path,
    _config_out: &Path,
    _options: TrainCnnOptions,
) -> Result<CnnModelMetadata, String> {
    Err("GPU training requires building with `--features gpu`; pass `--cpu` for the default local build".to_string())
}

pub fn save_safetensors<B: Backend>(model: &PicoDualCnn<B>, path: &Path) -> Result<(), String> {
    let mut store = SafetensorsStore::from_file(path).overwrite(true);
    model
        .save_into(&mut store)
        .map_err(|error| format!("failed to save {}: {error}", path.display()))
}

pub fn load_safetensors<B: Backend>(
    config: &PicoDualCnnConfig,
    path: &Path,
    device: &B::Device,
) -> Result<PicoDualCnn<B>, String> {
    let mut model = config.init::<B>(device);
    let mut store = SafetensorsStore::from_file(path);
    model
        .load_from(&mut store)
        .map_err(|error| format!("failed to load {}: {error}", path.display()))?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cnn::model::PicoDualCnnConfig;
    use crate::data::dataset::BranchSample;
    use burn::backend::NdArray;

    fn dual(id: &str, label: usize) -> DualSample {
        DualSample {
            sample_id: id.to_string(),
            label_index: label,
            one_d_values: vec![0.0; 3 * 64],
            two_d_values: vec![0.0; 28 * 28],
        }
    }

    #[test]
    fn balance_split_equalizes_labels_and_keeps_validation_per_label() {
        let mut samples = Vec::new();
        for label in 0..ALPHABET_LEN {
            for index in 0..5 {
                samples.push(dual(&format!("{label}-{index}"), label));
            }
        }
        samples.push(dual("extra-a", 0));
        let split = balanced_split(samples, 42).unwrap();
        assert_eq!(split.per_label_count, 5);
        assert_eq!(split.train.len(), 26 * 4);
        assert_eq!(split.validation.len(), 26);
    }

    #[test]
    fn balance_split_is_deterministic_for_seed() {
        let mut samples = Vec::new();
        for label in 0..ALPHABET_LEN {
            for index in 0..3 {
                samples.push(dual(&format!("{label}-{index}"), label));
            }
        }
        let left = balanced_split(samples.clone(), 7).unwrap();
        let right = balanced_split(samples, 7).unwrap();
        let left_ids = left
            .train
            .iter()
            .map(|sample| &sample.sample_id)
            .collect::<Vec<_>>();
        let right_ids = right
            .train
            .iter()
            .map(|sample| &sample.sample_id)
            .collect::<Vec<_>>();
        assert_eq!(left_ids, right_ids);
    }

    #[test]
    fn rejects_missing_label_samples() {
        let samples = (0..ALPHABET_LEN - 1)
            .flat_map(|label| {
                [
                    dual(&format!("{label}-0"), label),
                    dual(&format!("{label}-1"), label),
                ]
            })
            .collect::<Vec<_>>();
        assert!(balanced_split(samples, 42).is_err());
    }

    #[test]
    fn branch_sample_deserializes_label_index() {
        let sample: BranchSample = serde_json::from_str(
            r#"{"sample_id":"s","sample_label":"A","label_index":0,"extra_labels":[],"shape":[3,64],"values":[],"raw_point_count":2}"#,
        )
        .unwrap();
        assert_eq!(sample.label_index, 0);
    }

    #[test]
    fn safetensors_save_load_preserves_forward_shape() {
        type TestBackend = NdArray<f32>;

        let device = Default::default();
        let config = PicoDualCnnConfig::new();
        let model = config.init::<TestBackend>(&device);
        let path = std::env::temp_dir().join(format!(
            "crosswords-cnn-roundtrip-{}.safetensors",
            std::process::id()
        ));

        save_safetensors(&model, &path).unwrap();
        let loaded = load_safetensors::<TestBackend>(&config, &path, &device).unwrap();
        let _ = std::fs::remove_file(&path);

        let one_d = burn::tensor::Tensor::<TestBackend, 3>::zeros([1, 3, 64], &device);
        let two_d = burn::tensor::Tensor::<TestBackend, 4>::zeros([1, 1, 28, 28], &device);
        assert_eq!(loaded.forward(one_d, two_d).dims(), [1, 26]);
    }
}
