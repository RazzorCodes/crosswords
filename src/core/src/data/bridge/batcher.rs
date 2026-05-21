use std::collections::BTreeMap;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

use crate::data::dataset::BranchSample;
use crate::data::preproc::{CNN_1D_CHANNELS, RASTER_SIZE, RESAMPLED_POINTS};
use crate::types::ALPHABET_LEN;

#[derive(Clone, Debug)]
pub struct DualSample {
    pub sample_id: String,
    pub label_index: usize,
    pub one_d_values: Vec<f64>,
    pub two_d_values: Vec<f64>,
}

#[derive(Debug)]
pub struct CnnBatch<B: Backend> {
    pub one_d: Tensor<B, 3>,
    pub two_d: Tensor<B, 4>,
    pub labels: Tensor<B, 1, Int>,
}

fn valid_branch(sample: &BranchSample, expected_shape: &[usize]) -> Result<(), String> {
    if sample.label_index >= ALPHABET_LEN {
        return Err(format!(
            "sample {} has invalid label index {}",
            sample.sample_id, sample.label_index
        ));
    }
    if sample.shape != expected_shape {
        return Err(format!(
            "sample {} has shape {:?}; expected {:?}",
            sample.sample_id, sample.shape, expected_shape
        ));
    }
    let expected_len = expected_shape.iter().product::<usize>();
    if sample.values.len() != expected_len {
        return Err(format!(
            "sample {} has {} values; expected {expected_len}",
            sample.sample_id,
            sample.values.len()
        ));
    }
    if !sample.values.iter().all(|value| value.is_finite()) {
        return Err(format!(
            "sample {} contains non-finite values",
            sample.sample_id
        ));
    }
    if sample.raw_point_count < 2 {
        return Err(format!(
            "sample {} has fewer than 2 raw stroke points",
            sample.sample_id
        ));
    }
    Ok(())
}

pub fn pair_samples(
    one_d_samples: &[BranchSample],
    two_d_samples: &[BranchSample],
) -> Result<Vec<DualSample>, String> {
    let one_shape = [CNN_1D_CHANNELS, RESAMPLED_POINTS];
    let two_shape = [1, RASTER_SIZE, RASTER_SIZE];
    let mut one_by_id = BTreeMap::new();
    for sample in one_d_samples {
        valid_branch(sample, &one_shape)?;
        if one_by_id
            .insert(sample.sample_id.as_str(), sample)
            .is_some()
        {
            return Err(format!("duplicate 1D sample id {}", sample.sample_id));
        }
    }

    let mut paired = Vec::new();
    for two_d in two_d_samples {
        valid_branch(two_d, &two_shape)?;
        let Some(one_d) = one_by_id.remove(two_d.sample_id.as_str()) else {
            return Err(format!("missing 1D pair for sample {}", two_d.sample_id));
        };
        if one_d.label_index != two_d.label_index {
            return Err(format!(
                "sample {} has mismatched labels {} and {}",
                two_d.sample_id, one_d.label_index, two_d.label_index
            ));
        }
        paired.push(DualSample {
            sample_id: two_d.sample_id.clone(),
            label_index: two_d.label_index,
            one_d_values: one_d.values.clone(),
            two_d_values: two_d.values.clone(),
        });
    }

    if !one_by_id.is_empty() {
        let missing = one_by_id.keys().next().copied().unwrap_or("<unknown>");
        return Err(format!("missing 2D pair for sample {missing}"));
    }
    Ok(paired)
}

pub fn batch_for_1d<B: Backend>(
    samples: &[DualSample],
    device: &B::Device,
) -> Result<Tensor<B, 3>, String> {
    let mut values = Vec::with_capacity(samples.len() * CNN_1D_CHANNELS * RESAMPLED_POINTS);
    for sample in samples {
        if sample.one_d_values.len() != CNN_1D_CHANNELS * RESAMPLED_POINTS {
            return Err(format!(
                "sample {} has invalid 1D value count",
                sample.sample_id
            ));
        }
        if !sample.one_d_values.iter().all(|value| value.is_finite()) {
            return Err(format!(
                "sample {} contains non-finite 1D values",
                sample.sample_id
            ));
        }
        values.extend(sample.one_d_values.iter().map(|value| *value as f32));
    }
    Ok(Tensor::<B, 3>::from_data(
        TensorData::new(values, [samples.len(), CNN_1D_CHANNELS, RESAMPLED_POINTS]),
        device,
    ))
}

pub fn batch_for_2d<B: Backend>(
    samples: &[DualSample],
    device: &B::Device,
) -> Result<Tensor<B, 4>, String> {
    let mut values = Vec::with_capacity(samples.len() * RASTER_SIZE * RASTER_SIZE);
    for sample in samples {
        if sample.two_d_values.len() != RASTER_SIZE * RASTER_SIZE {
            return Err(format!(
                "sample {} has invalid 2D value count",
                sample.sample_id
            ));
        }
        if !sample.two_d_values.iter().all(|value| value.is_finite()) {
            return Err(format!(
                "sample {} contains non-finite 2D values",
                sample.sample_id
            ));
        }
        values.extend(sample.two_d_values.iter().map(|value| *value as f32));
    }
    Ok(Tensor::<B, 4>::from_data(
        TensorData::new(values, [samples.len(), 1, RASTER_SIZE, RASTER_SIZE]),
        device,
    ))
}

pub fn batch_labels<B: Backend>(
    samples: &[DualSample],
    device: &B::Device,
) -> Result<Tensor<B, 1, Int>, String> {
    let mut labels = Vec::with_capacity(samples.len());
    for sample in samples {
        if sample.label_index >= ALPHABET_LEN {
            return Err(format!(
                "sample {} has invalid label {}",
                sample.sample_id, sample.label_index
            ));
        }
        labels.push(sample.label_index as i32);
    }
    Ok(Tensor::<B, 1, Int>::from_data(
        TensorData::new(labels, [samples.len()]),
        device,
    ))
}

pub fn batch_dual<B: Backend>(
    samples: &[DualSample],
    device: &B::Device,
) -> Result<CnnBatch<B>, String> {
    if samples.is_empty() {
        return Err("cannot batch zero samples".to_string());
    }
    Ok(CnnBatch {
        one_d: batch_for_1d(samples, device)?,
        two_d: batch_for_2d(samples, device)?,
        labels: batch_labels(samples, device)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    fn branch(id: &str, label: usize, shape: Vec<usize>, values: Vec<f64>) -> BranchSample {
        BranchSample {
            sample_id: id.to_string(),
            sample_label: "A".to_string(),
            label_index: label,
            extra_labels: vec![],
            shape,
            values,
            raw_point_count: 2,
        }
    }

    fn dual(id: &str, label: usize) -> DualSample {
        DualSample {
            sample_id: id.to_string(),
            label_index: label,
            one_d_values: vec![0.0; 3 * 64],
            two_d_values: vec![0.0; 28 * 28],
        }
    }

    #[test]
    fn batch_shapes_are_expected() {
        let device = Default::default();
        let samples = vec![dual("s1", 0), dual("s2", 1)];
        let batch = batch_dual::<TestBackend>(&samples, &device).unwrap();
        assert_eq!(batch.one_d.dims(), [2, 3, 64]);
        assert_eq!(batch.two_d.dims(), [2, 1, 28, 28]);
        assert_eq!(batch.labels.dims(), [2]);
    }

    #[test]
    fn rejects_mismatched_pair_labels() {
        let one = vec![branch("s1", 0, vec![3, 64], vec![0.0; 3 * 64])];
        let two = vec![branch("s1", 1, vec![1, 28, 28], vec![0.0; 28 * 28])];
        assert!(pair_samples(&one, &two).is_err());
    }

    #[test]
    fn rejects_non_finite_values() {
        let mut values = vec![0.0; 3 * 64];
        values[3] = f64::NAN;
        let one = vec![branch("s1", 0, vec![3, 64], values)];
        let two = vec![branch("s1", 0, vec![1, 28, 28], vec![0.0; 28 * 28])];
        assert!(pair_samples(&one, &two).is_err());
    }

    #[test]
    fn rejects_one_point_raw_strokes() {
        let mut one = branch("s1", 0, vec![3, 64], vec![0.0; 3 * 64]);
        one.raw_point_count = 1;
        let two = vec![branch("s1", 0, vec![1, 28, 28], vec![0.0; 28 * 28])];
        assert!(pair_samples(&[one], &two).is_err());
    }
}
