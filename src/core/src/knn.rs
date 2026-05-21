use crate::types::{euclidean_distance, label_to_index, ALPHABET_LEN, FEATURE_COUNT};
use std::cmp::Ordering;

pub fn predict(
    features: &[f64],
    cache_features: &[f64],
    cache_labels: &[u8],
    cache_size: usize,
    k: usize,
    far_neighbor_distance: f64,
    out: &mut [f64],
) -> usize {
    out.fill(0.0);
    if cache_size == 0 {
        return 0;
    }

    let mut distances = Vec::with_capacity(cache_size);
    for index in 0..cache_size {
        if let Some(label_index) = label_to_index(cache_labels[index]) {
            let start = index * FEATURE_COUNT;
            let distance =
                euclidean_distance(features, &cache_features[start..start + FEATURE_COUNT]);
            distances.push((label_index, distance));
        }
    }
    distances.sort_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(Ordering::Equal));

    let nearest: Vec<(usize, f64)> = distances.into_iter().take(k).collect();
    if nearest.is_empty() || nearest[0].1 > far_neighbor_distance {
        return 0;
    }

    let mut total_weight = 0.0_f64;
    for (label_index, distance) in nearest {
        let weight = 1.0 / distance.max(1e-6);
        total_weight += weight;
        out[label_index] += weight;
    }

    if total_weight <= 0.0 {
        return 0;
    }

    let mut non_zero = 0_usize;
    for value in out.iter_mut().take(ALPHABET_LEN) {
        if *value > 0.0 {
            *value /= total_weight;
            non_zero += 1;
        }
    }
    non_zero
}
