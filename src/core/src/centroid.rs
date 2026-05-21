use crate::types::{euclidean_distance, label_to_index, ALPHABET_LEN, FEATURE_COUNT};

pub fn train(
    sample_features: &[f64],
    sample_labels: &[u8],
    sample_count: usize,
    ready_labels: &[u8],
    out_centroids: &mut [f64],
    out_counts: &mut [u32],
) -> usize {
    out_centroids.fill(0.0);
    out_counts.fill(0);

    for (ready_index, ready_label) in ready_labels.iter().enumerate() {
        let Some(label_index) = label_to_index(*ready_label) else {
            continue;
        };
        let mut count = 0_u32;
        for sample_index in 0..sample_count {
            if sample_labels[sample_index] as usize != label_index {
                continue;
            }
            let feature_offset = sample_index * FEATURE_COUNT;
            let centroid_offset = ready_index * FEATURE_COUNT;
            for feature_index in 0..FEATURE_COUNT {
                out_centroids[centroid_offset + feature_index] +=
                    sample_features[feature_offset + feature_index];
            }
            count += 1;
        }

        if count > 0 {
            let centroid_offset = ready_index * FEATURE_COUNT;
            for feature_index in 0..FEATURE_COUNT {
                out_centroids[centroid_offset + feature_index] /= count as f64;
            }
            out_counts[ready_index] = count;
        }
    }

    ready_labels.len()
}

pub fn predict(
    centroids: &[f64],
    centroid_labels: &[u8],
    centroid_count: usize,
    features: &[f64],
    out: &mut [f64],
) {
    out.fill(0.0);

    if centroid_count == 0 {
        let uniform = 1.0 / ALPHABET_LEN as f64;
        out.fill(uniform);
        return;
    }

    let mut scores = Vec::with_capacity(centroid_count);
    let mut total = 0.0_f64;
    for index in 0..centroid_count {
        let offset = index * FEATURE_COUNT;
        let score =
            (-euclidean_distance(features, &centroids[offset..offset + FEATURE_COUNT])).exp();
        scores.push(score);
        total += score;
    }

    if total <= 0.0 {
        let uniform = 1.0 / centroid_count as f64;
        for label in centroid_labels {
            if let Some(label_index) = label_to_index(*label) {
                out[label_index] = uniform;
            }
        }
        return;
    }

    for (index, score) in scores.into_iter().enumerate() {
        if let Some(label_index) = label_to_index(centroid_labels[index]) {
            out[label_index] = score / total;
        }
    }
}
