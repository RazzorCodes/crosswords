use crate::types::{label_to_index, ALPHABET_LEN, FEATURE_COUNT};

const DEFAULT_TOLERANCE: f64 = 1e-3;
const DEFAULT_MAX_PASSES: usize = 12;
const MIN_ALPHA: f64 = 1e-8;

#[derive(Debug)]
pub struct SvmExport<'a> {
    pub labels: &'a mut [u8],
    pub biases: &'a mut [f64],
    pub starts: &'a mut [u32],
    pub counts: &'a mut [u32],
    pub coefficients: &'a mut [f64],
    pub support_features: &'a mut [f64],
    pub feature_mean: &'a mut [f64],
    pub feature_std: &'a mut [f64],
}

fn feature(sample_features: &[f64], sample_index: usize, feature_index: usize) -> f64 {
    sample_features[sample_index * FEATURE_COUNT + feature_index]
}

fn rbf_kernel(a: &[f64], b: &[f64], gamma: f64) -> f64 {
    let mut sum = 0.0_f64;
    for index in 0..FEATURE_COUNT {
        let delta = a[index] - b[index];
        sum += delta * delta;
    }
    (-gamma * sum).exp()
}

fn compute_feature_stats(
    sample_features: &[f64],
    sample_count: usize,
    mean: &mut [f64],
    std: &mut [f64],
) {
    mean.fill(0.0);
    std.fill(0.0);
    if sample_count == 0 {
        std.fill(1.0);
        return;
    }

    for sample_index in 0..sample_count {
        for feature_index in 0..FEATURE_COUNT {
            mean[feature_index] += feature(sample_features, sample_index, feature_index);
        }
    }
    for value in mean.iter_mut().take(FEATURE_COUNT) {
        *value /= sample_count as f64;
    }

    for sample_index in 0..sample_count {
        for feature_index in 0..FEATURE_COUNT {
            let delta = feature(sample_features, sample_index, feature_index) - mean[feature_index];
            std[feature_index] += delta * delta;
        }
    }
    for value in std.iter_mut().take(FEATURE_COUNT) {
        *value = (*value / sample_count as f64).sqrt().max(1e-6);
    }
}

fn standardize_samples(
    sample_features: &[f64],
    sample_count: usize,
    mean: &[f64],
    std: &[f64],
) -> Vec<f64> {
    let mut result = vec![0.0; sample_count * FEATURE_COUNT];
    for sample_index in 0..sample_count {
        for feature_index in 0..FEATURE_COUNT {
            result[sample_index * FEATURE_COUNT + feature_index] =
                (feature(sample_features, sample_index, feature_index) - mean[feature_index])
                    / std[feature_index];
        }
    }
    result
}

fn sample_slice(features: &[f64], sample_index: usize) -> &[f64] {
    let offset = sample_index * FEATURE_COUNT;
    &features[offset..offset + FEATURE_COUNT]
}

fn decision_for_index(
    sample_index: usize,
    features: &[f64],
    labels: &[f64],
    alphas: &[f64],
    bias: f64,
    gamma: f64,
) -> f64 {
    let query = sample_slice(features, sample_index);
    let mut sum = bias;
    for support_index in 0..labels.len() {
        if alphas[support_index] <= MIN_ALPHA {
            continue;
        }
        sum += alphas[support_index]
            * labels[support_index]
            * rbf_kernel(sample_slice(features, support_index), query, gamma);
    }
    sum
}

fn train_binary_smo(features: &[f64], labels: &[f64], c: f64, gamma: f64) -> (Vec<f64>, f64) {
    let sample_count = labels.len();
    let mut alphas = vec![0.0_f64; sample_count];
    let mut bias = 0.0_f64;
    let mut passes = 0_usize;
    let c = c.max(1e-6);

    while passes < DEFAULT_MAX_PASSES {
        let mut changed = 0_usize;
        for i in 0..sample_count {
            let error_i = decision_for_index(i, features, labels, &alphas, bias, gamma) - labels[i];
            if !((labels[i] * error_i < -DEFAULT_TOLERANCE && alphas[i] < c)
                || (labels[i] * error_i > DEFAULT_TOLERANCE && alphas[i] > 0.0))
            {
                continue;
            }

            // Deterministic second variable selection keeps browser retraining reproducible.
            let j = (i + 1 + (passes % sample_count.max(1))) % sample_count;
            if i == j {
                continue;
            }

            let error_j = decision_for_index(j, features, labels, &alphas, bias, gamma) - labels[j];
            let old_i = alphas[i];
            let old_j = alphas[j];

            let (low, high) = if labels[i] != labels[j] {
                ((old_j - old_i).max(0.0), (c + old_j - old_i).min(c))
            } else {
                ((old_i + old_j - c).max(0.0), (old_i + old_j).min(c))
            };
            if (high - low).abs() < 1e-12 {
                continue;
            }

            let kii = rbf_kernel(sample_slice(features, i), sample_slice(features, i), gamma);
            let kjj = rbf_kernel(sample_slice(features, j), sample_slice(features, j), gamma);
            let kij = rbf_kernel(sample_slice(features, i), sample_slice(features, j), gamma);
            let eta = 2.0 * kij - kii - kjj;
            if eta >= 0.0 {
                continue;
            }

            alphas[j] -= labels[j] * (error_i - error_j) / eta;
            alphas[j] = alphas[j].clamp(low, high);
            if (alphas[j] - old_j).abs() < 1e-5 {
                continue;
            }
            alphas[i] += labels[i] * labels[j] * (old_j - alphas[j]);

            let b1 = bias
                - error_i
                - labels[i] * (alphas[i] - old_i) * kii
                - labels[j] * (alphas[j] - old_j) * kij;
            let b2 = bias
                - error_j
                - labels[i] * (alphas[i] - old_i) * kij
                - labels[j] * (alphas[j] - old_j) * kjj;

            bias = if alphas[i] > 0.0 && alphas[i] < c {
                b1
            } else if alphas[j] > 0.0 && alphas[j] < c {
                b2
            } else {
                (b1 + b2) * 0.5
            };
            changed += 1;
        }

        if changed == 0 {
            passes += 1;
        } else {
            passes = 0;
        }
    }

    (alphas, bias)
}

pub fn train_one_vs_rest(
    sample_features: &[f64],
    sample_labels: &[u8],
    sample_count: usize,
    ready_labels: &[u8],
    c: f64,
    gamma: f64,
    export: &mut SvmExport<'_>,
) -> usize {
    export.labels.fill(255);
    export.biases.fill(0.0);
    export.starts.fill(0);
    export.counts.fill(0);
    export.coefficients.fill(0.0);
    export.support_features.fill(0.0);
    compute_feature_stats(
        sample_features,
        sample_count,
        export.feature_mean,
        export.feature_std,
    );

    if sample_count == 0 || ready_labels.len() < 2 {
        return 0;
    }

    let standardized = standardize_samples(
        sample_features,
        sample_count,
        export.feature_mean,
        export.feature_std,
    );
    let mut total_support_count = 0_usize;

    for (classifier_index, ready_label) in ready_labels.iter().enumerate() {
        let Some(positive_label) = label_to_index(*ready_label) else {
            continue;
        };
        let binary_labels: Vec<f64> = sample_labels
            .iter()
            .take(sample_count)
            .map(|label| {
                if *label as usize == positive_label {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();

        if !binary_labels.iter().any(|value| *value > 0.0)
            || !binary_labels.iter().any(|value| *value < 0.0)
        {
            continue;
        }

        let (alphas, bias) = train_binary_smo(&standardized, &binary_labels, c, gamma);
        export.labels[classifier_index] = *ready_label;
        export.biases[classifier_index] = bias;
        export.starts[classifier_index] = total_support_count as u32;

        for sample_index in 0..sample_count {
            if alphas[sample_index] <= MIN_ALPHA {
                continue;
            }
            let support_offset = total_support_count * FEATURE_COUNT;
            if support_offset + FEATURE_COUNT > export.support_features.len()
                || total_support_count >= export.coefficients.len()
            {
                break;
            }
            export.coefficients[total_support_count] =
                alphas[sample_index] * binary_labels[sample_index];
            export.support_features[support_offset..support_offset + FEATURE_COUNT]
                .copy_from_slice(sample_slice(&standardized, sample_index));
            total_support_count += 1;
        }
        export.counts[classifier_index] =
            (total_support_count - export.starts[classifier_index] as usize) as u32;
    }

    total_support_count
}

fn standardize_feature(
    features: &[f64],
    mean: &[f64],
    std: &[f64],
    out: &mut [f64; FEATURE_COUNT],
) {
    for index in 0..FEATURE_COUNT {
        out[index] = (features[index] - mean[index]) / std[index].max(1e-6);
    }
}

pub fn predict_one_vs_rest(
    features: &[f64],
    labels: &[u8],
    biases: &[f64],
    starts: &[u32],
    counts: &[u32],
    coefficients: &[f64],
    support_features: &[f64],
    classifier_count: usize,
    gamma: f64,
    feature_mean: &[f64],
    feature_std: &[f64],
    out: &mut [f64],
) {
    out.fill(0.0);
    if classifier_count == 0 {
        out.fill(1.0 / ALPHABET_LEN as f64);
        return;
    }

    let mut query = [0.0_f64; FEATURE_COUNT];
    standardize_feature(features, feature_mean, feature_std, &mut query);
    let mut margins = Vec::with_capacity(classifier_count);

    for classifier_index in 0..classifier_count {
        let start = starts[classifier_index] as usize;
        let count = counts[classifier_index] as usize;
        let mut margin = biases[classifier_index];
        for support_index in start..start + count {
            let support_offset = support_index * FEATURE_COUNT;
            if support_offset + FEATURE_COUNT > support_features.len()
                || support_index >= coefficients.len()
            {
                continue;
            }
            margin += coefficients[support_index]
                * rbf_kernel(
                    &support_features[support_offset..support_offset + FEATURE_COUNT],
                    &query,
                    gamma,
                );
        }
        margins.push(margin);
    }

    let max_margin = margins.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut total = 0.0_f64;
    for (classifier_index, margin) in margins.into_iter().enumerate() {
        let score = (margin - max_margin).exp();
        if let Some(label_index) = label_to_index(labels[classifier_index]) {
            out[label_index] = score;
            total += score;
        }
    }

    if total <= 0.0 || !total.is_finite() {
        let uniform = 1.0 / classifier_count as f64;
        for label in labels.iter().take(classifier_count) {
            if let Some(label_index) = label_to_index(*label) {
                out[label_index] = uniform;
            }
        }
        return;
    }

    for value in out.iter_mut() {
        if *value > 0.0 {
            *value /= total;
        }
    }
}
