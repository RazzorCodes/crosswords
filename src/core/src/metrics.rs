use crate::centroid;
use crate::types::{label_to_index, top_label_from_probabilities, ALPHABET_LEN, FEATURE_COUNT};

pub fn evaluate_centroid_snapshot(
    holdout_features: &[f64],
    holdout_labels: &[u8],
    holdout_acceptances: &[u8],
    holdout_count: usize,
    centroids: &[f64],
    centroid_labels: &[u8],
    centroid_count: usize,
    out_metrics: &mut [f64],
) {
    out_metrics.fill(0.0);
    if holdout_count == 0 || centroid_count == 0 {
        return;
    }

    let mut user_total = 0_u32;
    let mut user_correct = 0_u32;
    let mut implicit_total = 0_u32;
    let mut implicit_correct = 0_u32;
    let mut overall_total = 0_u32;
    let mut overall_correct = 0_u32;
    let mut probabilities = [0.0_f64; ALPHABET_LEN];

    for sample_index in 0..holdout_count {
        let feature_offset = sample_index * FEATURE_COUNT;
        centroid::predict(
            centroids,
            centroid_labels,
            centroid_count,
            &holdout_features[feature_offset..feature_offset + FEATURE_COUNT],
            &mut probabilities,
        );
        let predicted = top_label_from_probabilities(&probabilities);
        let actual = label_to_index(holdout_labels[sample_index]);
        let correct = predicted.is_some() && predicted == actual;
        overall_total += 1;
        if correct {
            overall_correct += 1;
        }
        if holdout_acceptances[sample_index] == 1 {
            user_total += 1;
            if correct {
                user_correct += 1;
            }
        } else {
            implicit_total += 1;
            if correct {
                implicit_correct += 1;
            }
        }
    }

    out_metrics[0] = if user_total > 0 {
        user_correct as f64 / user_total as f64
    } else {
        0.0
    };
    out_metrics[1] = if implicit_total > 0 {
        implicit_correct as f64 / implicit_total as f64
    } else {
        0.0
    };
    out_metrics[2] = if overall_total > 0 {
        overall_correct as f64 / overall_total as f64
    } else {
        0.0
    };
}
