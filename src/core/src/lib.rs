use std::alloc::{alloc as alloc_bytes, dealloc as dealloc_bytes, Layout};
use std::slice;

#[path = "algorithm/centroid/centroid.rs"]
mod centroid;
pub mod data;
mod dataset;
#[path = "algorithm/centroid/features.rs"]
mod features;
#[path = "algorithm/k-nn/knn.rs"]
mod knn;
#[path = "metrics/metrics.rs"]
mod metrics;
#[path = "algorithm/svm/svm.rs"]
mod svm;
mod types;

use types::{ALPHABET_LEN, FEATURE_COUNT};

#[no_mangle]
pub extern "C" fn alloc(size: usize) -> *mut u8 {
    if size == 0 {
        return std::ptr::null_mut();
    }
    let layout = Layout::from_size_align(size, 8).unwrap();
    unsafe { alloc_bytes(layout) }
}

#[no_mangle]
pub extern "C" fn dealloc(ptr: *mut u8, size: usize) {
    if ptr.is_null() || size == 0 {
        return;
    }
    let layout = Layout::from_size_align(size, 8).unwrap();
    unsafe { dealloc_bytes(ptr, layout) }
}

#[no_mangle]
pub extern "C" fn extract_features(
    points_ptr: *const f64,
    point_count: usize,
    stroke_lengths_ptr: *const u32,
    stroke_count: usize,
    out_ptr: *mut f64,
) {
    if points_ptr.is_null() || stroke_lengths_ptr.is_null() || out_ptr.is_null() {
        return;
    }

    let features = unsafe {
        let points = slice::from_raw_parts(points_ptr, point_count * 3);
        let stroke_lengths = slice::from_raw_parts(stroke_lengths_ptr, stroke_count);
        let strokes = features::decode_strokes(points, stroke_lengths);
        features::extract_features(&strokes)
    };

    unsafe {
        slice::from_raw_parts_mut(out_ptr, FEATURE_COUNT).copy_from_slice(&features);
    }
}

#[no_mangle]
pub extern "C" fn knn_predict(
    features_ptr: *const f64,
    cache_features_ptr: *const f64,
    cache_labels_ptr: *const u8,
    cache_size: usize,
    k: usize,
    far_neighbor_distance: f64,
    out_ptr: *mut f64,
) -> usize {
    if features_ptr.is_null() || out_ptr.is_null() {
        return 0;
    }

    let features = unsafe { slice::from_raw_parts(features_ptr, FEATURE_COUNT) };
    let out = unsafe { slice::from_raw_parts_mut(out_ptr, ALPHABET_LEN) };

    if cache_size == 0 || cache_features_ptr.is_null() || cache_labels_ptr.is_null() {
        out.fill(0.0);
        return 0;
    }

    let cache_features =
        unsafe { slice::from_raw_parts(cache_features_ptr, cache_size * FEATURE_COUNT) };
    let cache_labels = unsafe { slice::from_raw_parts(cache_labels_ptr, cache_size) };
    knn::predict(
        features,
        cache_features,
        cache_labels,
        cache_size,
        k,
        far_neighbor_distance,
        out,
    )
}

#[no_mangle]
pub extern "C" fn train_centroid_classifier(
    sample_features_ptr: *const f64,
    sample_labels_ptr: *const u8,
    sample_count: usize,
    ready_labels_ptr: *const u8,
    ready_count: usize,
    out_centroids_ptr: *mut f64,
    out_counts_ptr: *mut u32,
) -> usize {
    if sample_count == 0
        || ready_count == 0
        || sample_features_ptr.is_null()
        || sample_labels_ptr.is_null()
        || ready_labels_ptr.is_null()
        || out_centroids_ptr.is_null()
        || out_counts_ptr.is_null()
    {
        return 0;
    }

    let sample_features =
        unsafe { slice::from_raw_parts(sample_features_ptr, sample_count * FEATURE_COUNT) };
    let sample_labels = unsafe { slice::from_raw_parts(sample_labels_ptr, sample_count) };
    let ready_labels = unsafe { slice::from_raw_parts(ready_labels_ptr, ready_count) };
    let out_centroids =
        unsafe { slice::from_raw_parts_mut(out_centroids_ptr, ready_count * FEATURE_COUNT) };
    let out_counts = unsafe { slice::from_raw_parts_mut(out_counts_ptr, ready_count) };
    centroid::train(
        sample_features,
        sample_labels,
        sample_count,
        ready_labels,
        out_centroids,
        out_counts,
    )
}

#[no_mangle]
pub extern "C" fn predict_centroid_classifier(
    centroids_ptr: *const f64,
    centroid_labels_ptr: *const u8,
    centroid_count: usize,
    features_ptr: *const f64,
    out_probs_ptr: *mut f64,
) {
    if centroids_ptr.is_null()
        || centroid_labels_ptr.is_null()
        || features_ptr.is_null()
        || out_probs_ptr.is_null()
    {
        return;
    }

    let centroids = unsafe { slice::from_raw_parts(centroids_ptr, centroid_count * FEATURE_COUNT) };
    let centroid_labels = unsafe { slice::from_raw_parts(centroid_labels_ptr, centroid_count) };
    let features = unsafe { slice::from_raw_parts(features_ptr, FEATURE_COUNT) };
    let out = unsafe { slice::from_raw_parts_mut(out_probs_ptr, ALPHABET_LEN) };
    centroid::predict(centroids, centroid_labels, centroid_count, features, out);
}

#[no_mangle]
pub extern "C" fn train_svm_classifier(
    sample_features_ptr: *const f64,
    sample_labels_ptr: *const u8,
    sample_count: usize,
    ready_labels_ptr: *const u8,
    ready_count: usize,
    c: f64,
    gamma: f64,
    out_labels_ptr: *mut u8,
    out_biases_ptr: *mut f64,
    out_starts_ptr: *mut u32,
    out_counts_ptr: *mut u32,
    out_coefficients_ptr: *mut f64,
    out_support_features_ptr: *mut f64,
    out_feature_mean_ptr: *mut f64,
    out_feature_std_ptr: *mut f64,
) -> usize {
    if sample_count == 0
        || ready_count < 2
        || sample_features_ptr.is_null()
        || sample_labels_ptr.is_null()
        || ready_labels_ptr.is_null()
        || out_labels_ptr.is_null()
        || out_biases_ptr.is_null()
        || out_starts_ptr.is_null()
        || out_counts_ptr.is_null()
        || out_coefficients_ptr.is_null()
        || out_support_features_ptr.is_null()
        || out_feature_mean_ptr.is_null()
        || out_feature_std_ptr.is_null()
    {
        return 0;
    }

    let max_support_count = sample_count * ready_count;
    let sample_features =
        unsafe { slice::from_raw_parts(sample_features_ptr, sample_count * FEATURE_COUNT) };
    let sample_labels = unsafe { slice::from_raw_parts(sample_labels_ptr, sample_count) };
    let ready_labels = unsafe { slice::from_raw_parts(ready_labels_ptr, ready_count) };
    let mut export = svm::SvmExport {
        labels: unsafe { slice::from_raw_parts_mut(out_labels_ptr, ready_count) },
        biases: unsafe { slice::from_raw_parts_mut(out_biases_ptr, ready_count) },
        starts: unsafe { slice::from_raw_parts_mut(out_starts_ptr, ready_count) },
        counts: unsafe { slice::from_raw_parts_mut(out_counts_ptr, ready_count) },
        coefficients: unsafe { slice::from_raw_parts_mut(out_coefficients_ptr, max_support_count) },
        support_features: unsafe {
            slice::from_raw_parts_mut(out_support_features_ptr, max_support_count * FEATURE_COUNT)
        },
        feature_mean: unsafe { slice::from_raw_parts_mut(out_feature_mean_ptr, FEATURE_COUNT) },
        feature_std: unsafe { slice::from_raw_parts_mut(out_feature_std_ptr, FEATURE_COUNT) },
    };

    svm::train_one_vs_rest(
        sample_features,
        sample_labels,
        sample_count,
        ready_labels,
        c,
        gamma,
        &mut export,
    )
}

#[no_mangle]
pub extern "C" fn predict_svm_classifier(
    features_ptr: *const f64,
    labels_ptr: *const u8,
    biases_ptr: *const f64,
    starts_ptr: *const u32,
    counts_ptr: *const u32,
    coefficients_ptr: *const f64,
    support_features_ptr: *const f64,
    classifier_count: usize,
    total_support_count: usize,
    gamma: f64,
    feature_mean_ptr: *const f64,
    feature_std_ptr: *const f64,
    out_probs_ptr: *mut f64,
) {
    if features_ptr.is_null()
        || labels_ptr.is_null()
        || biases_ptr.is_null()
        || starts_ptr.is_null()
        || counts_ptr.is_null()
        || coefficients_ptr.is_null()
        || support_features_ptr.is_null()
        || feature_mean_ptr.is_null()
        || feature_std_ptr.is_null()
        || out_probs_ptr.is_null()
    {
        return;
    }

    let features = unsafe { slice::from_raw_parts(features_ptr, FEATURE_COUNT) };
    let labels = unsafe { slice::from_raw_parts(labels_ptr, classifier_count) };
    let biases = unsafe { slice::from_raw_parts(biases_ptr, classifier_count) };
    let starts = unsafe { slice::from_raw_parts(starts_ptr, classifier_count) };
    let counts = unsafe { slice::from_raw_parts(counts_ptr, classifier_count) };
    let coefficients = unsafe { slice::from_raw_parts(coefficients_ptr, total_support_count) };
    let support_features =
        unsafe { slice::from_raw_parts(support_features_ptr, total_support_count * FEATURE_COUNT) };
    let feature_mean = unsafe { slice::from_raw_parts(feature_mean_ptr, FEATURE_COUNT) };
    let feature_std = unsafe { slice::from_raw_parts(feature_std_ptr, FEATURE_COUNT) };
    let out = unsafe { slice::from_raw_parts_mut(out_probs_ptr, ALPHABET_LEN) };

    svm::predict_one_vs_rest(
        features,
        labels,
        biases,
        starts,
        counts,
        coefficients,
        support_features,
        classifier_count,
        gamma,
        feature_mean,
        feature_std,
        out,
    );
}

#[no_mangle]
pub extern "C" fn compute_letter_stats(
    labels_ptr: *const u8,
    acceptances_ptr: *const u8,
    sample_count: usize,
    out_counts_ptr: *mut u32,
    out_ready_ptr: *mut u8,
    out_priority_ptr: *mut u8,
) -> i32 {
    if out_counts_ptr.is_null() || out_ready_ptr.is_null() || out_priority_ptr.is_null() {
        return -1;
    }

    let out_counts = unsafe { slice::from_raw_parts_mut(out_counts_ptr, ALPHABET_LEN * 2) };
    let out_ready = unsafe { slice::from_raw_parts_mut(out_ready_ptr, ALPHABET_LEN) };
    let out_priority = unsafe { slice::from_raw_parts_mut(out_priority_ptr, ALPHABET_LEN) };
    if sample_count == 0 || labels_ptr.is_null() || acceptances_ptr.is_null() {
        out_counts.fill(0);
        out_ready.fill(0);
        out_priority.fill(0);
        return -1;
    }

    let labels = unsafe { slice::from_raw_parts(labels_ptr, sample_count) };
    let acceptances = unsafe { slice::from_raw_parts(acceptances_ptr, sample_count) };
    dataset::compute_letter_stats(labels, acceptances, out_counts, out_ready, out_priority)
}

#[no_mangle]
pub extern "C" fn build_balanced_dataset(
    labels_ptr: *const u8,
    acceptances_ptr: *const u8,
    created_at_ptr: *const f64,
    sample_count: usize,
    out_training_mask_ptr: *mut u8,
    out_holdout_mask_ptr: *mut u8,
    out_ready_ptr: *mut u8,
) -> u32 {
    if out_training_mask_ptr.is_null() || out_holdout_mask_ptr.is_null() || out_ready_ptr.is_null()
    {
        return 0;
    }

    let out_training_mask =
        unsafe { slice::from_raw_parts_mut(out_training_mask_ptr, sample_count) };
    let out_holdout_mask = unsafe { slice::from_raw_parts_mut(out_holdout_mask_ptr, sample_count) };
    let out_ready = unsafe { slice::from_raw_parts_mut(out_ready_ptr, ALPHABET_LEN) };
    if sample_count == 0
        || labels_ptr.is_null()
        || acceptances_ptr.is_null()
        || created_at_ptr.is_null()
    {
        out_training_mask.fill(0);
        out_holdout_mask.fill(0);
        out_ready.fill(0);
        return 0;
    }

    let labels = unsafe { slice::from_raw_parts(labels_ptr, sample_count) };
    let acceptances = unsafe { slice::from_raw_parts(acceptances_ptr, sample_count) };
    let created_at = unsafe { slice::from_raw_parts(created_at_ptr, sample_count) };
    dataset::build_balanced_dataset(
        labels,
        acceptances,
        created_at,
        out_training_mask,
        out_holdout_mask,
        out_ready,
    )
}

#[no_mangle]
pub extern "C" fn evaluate_snapshot(
    holdout_features_ptr: *const f64,
    holdout_labels_ptr: *const u8,
    holdout_acceptances_ptr: *const u8,
    holdout_count: usize,
    centroids_ptr: *const f64,
    centroid_labels_ptr: *const u8,
    centroid_count: usize,
    out_metrics_ptr: *mut f64,
) {
    if out_metrics_ptr.is_null() {
        return;
    }

    let out_metrics = unsafe { slice::from_raw_parts_mut(out_metrics_ptr, 3) };
    if holdout_count == 0
        || centroid_count == 0
        || holdout_features_ptr.is_null()
        || holdout_labels_ptr.is_null()
        || holdout_acceptances_ptr.is_null()
        || centroids_ptr.is_null()
        || centroid_labels_ptr.is_null()
    {
        out_metrics.fill(0.0);
        return;
    }

    let holdout_features =
        unsafe { slice::from_raw_parts(holdout_features_ptr, holdout_count * FEATURE_COUNT) };
    let holdout_labels = unsafe { slice::from_raw_parts(holdout_labels_ptr, holdout_count) };
    let holdout_acceptances =
        unsafe { slice::from_raw_parts(holdout_acceptances_ptr, holdout_count) };
    let centroids = unsafe { slice::from_raw_parts(centroids_ptr, centroid_count * FEATURE_COUNT) };
    let centroid_labels = unsafe { slice::from_raw_parts(centroid_labels_ptr, centroid_count) };
    metrics::evaluate_centroid_snapshot(
        holdout_features,
        holdout_labels,
        holdout_acceptances,
        holdout_count,
        centroids,
        centroid_labels,
        centroid_count,
        out_metrics,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Point;

    fn point(x: f64, y: f64, t: f64) -> Point {
        Point { x, y, t }
    }

    fn feature_vec(value: f64) -> [f64; FEATURE_COUNT] {
        [value; FEATURE_COUNT]
    }

    fn flatten_features(features: &[[f64; FEATURE_COUNT]]) -> Vec<f64> {
        features
            .iter()
            .flat_map(|item| item.iter().copied())
            .collect()
    }

    #[test]
    fn extract_features_is_deterministic_and_finite() {
        let strokes = vec![
            vec![
                point(0.0, 0.0, 0.0),
                point(1.0, 1.0, 1.0),
                point(2.0, 1.0, 2.0),
            ],
            vec![point(2.0, 1.0, 3.0), point(1.0, 0.0, 4.0)],
        ];

        let first = features::extract_features(&strokes);
        let second = features::extract_features(&strokes);

        assert_eq!(first.len(), FEATURE_COUNT);
        assert_eq!(first, second);
        assert!(first.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn knn_predict_prefers_nearest_neighbors() {
        let query = feature_vec(0.0);
        let cache = flatten_features(&[feature_vec(0.0), feature_vec(0.2), feature_vec(5.0)]);
        let labels = [0_u8, 0_u8, 1_u8];
        let mut out = [0.0_f64; ALPHABET_LEN];

        let non_zero = knn_predict(
            query.as_ptr(),
            cache.as_ptr(),
            labels.as_ptr(),
            labels.len(),
            2,
            10.0,
            out.as_mut_ptr(),
        );

        assert!(non_zero > 0);
        assert!(out[0] > out[1]);
        assert!(out[0] > 0.0);
    }

    #[test]
    fn svm_training_and_prediction_separates_simple_clusters() {
        let samples = flatten_features(&[
            feature_vec(1.0),
            feature_vec(1.2),
            feature_vec(9.0),
            feature_vec(9.3),
        ]);
        let sample_labels = [0_u8, 0_u8, 1_u8, 1_u8];
        let ready_labels = [0_u8, 1_u8];
        let max_support = sample_labels.len() * ready_labels.len();
        let mut labels = [255_u8; 2];
        let mut biases = [0.0_f64; 2];
        let mut starts = [0_u32; 2];
        let mut counts = [0_u32; 2];
        let mut coefficients = vec![0.0_f64; max_support];
        let mut support_features = vec![0.0_f64; max_support * FEATURE_COUNT];
        let mut mean = [0.0_f64; FEATURE_COUNT];
        let mut std = [0.0_f64; FEATURE_COUNT];

        let total_support = train_svm_classifier(
            samples.as_ptr(),
            sample_labels.as_ptr(),
            sample_labels.len(),
            ready_labels.as_ptr(),
            ready_labels.len(),
            10.0,
            1.0 / FEATURE_COUNT as f64,
            labels.as_mut_ptr(),
            biases.as_mut_ptr(),
            starts.as_mut_ptr(),
            counts.as_mut_ptr(),
            coefficients.as_mut_ptr(),
            support_features.as_mut_ptr(),
            mean.as_mut_ptr(),
            std.as_mut_ptr(),
        );
        assert!(total_support > 0);

        let query = feature_vec(9.1);
        let mut out = [0.0_f64; ALPHABET_LEN];
        predict_svm_classifier(
            query.as_ptr(),
            labels.as_ptr(),
            biases.as_ptr(),
            starts.as_ptr(),
            counts.as_ptr(),
            coefficients.as_ptr(),
            support_features.as_ptr(),
            ready_labels.len(),
            total_support,
            1.0 / FEATURE_COUNT as f64,
            mean.as_ptr(),
            std.as_ptr(),
            out.as_mut_ptr(),
        );
        assert!(
            out[1] > out[0],
            "expected B-like cluster to beat A-like cluster: {out:?}"
        );
    }
}
