use std::cmp::Ordering;
use std::alloc::{alloc as alloc_bytes, dealloc as dealloc_bytes, Layout};
use std::slice;

const ALPHABET_LEN: usize = 26;
const FEATURE_COUNT: usize = 30;
const MIN_READY_SAMPLES_PER_LETTER: usize = 5;
const MIN_READY_USER_INPUTTED_PER_LETTER: usize = 1;
const HOLDOUT_FRACTION: f64 = 0.2;

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

#[derive(Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
    t: f64,
}

fn safe_order(left: f64, right: f64) -> Ordering {
    right.partial_cmp(&left).unwrap_or(Ordering::Equal)
}

unsafe fn points_from_raw<'a>(
    points_ptr: *const f64,
    point_count: usize,
) -> &'a [f64] {
    slice::from_raw_parts(points_ptr, point_count * 3)
}

unsafe fn stroke_lengths_from_raw<'a>(
    lengths_ptr: *const u32,
    stroke_count: usize,
) -> &'a [u32] {
    slice::from_raw_parts(lengths_ptr, stroke_count)
}

fn decode_strokes(points: &[f64], stroke_lengths: &[u32]) -> Vec<Vec<Point>> {
    let mut offset = 0_usize;
    let mut strokes = Vec::with_capacity(stroke_lengths.len());
    for &length in stroke_lengths {
        let length = length as usize;
        let mut stroke = Vec::with_capacity(length);
        for _ in 0..length {
            let base = offset * 3;
            if base + 2 >= points.len() {
                break;
            }
            stroke.push(Point {
                x: points[base],
                y: points[base + 1],
                t: points[base + 2],
            });
            offset += 1;
        }
        strokes.push(stroke);
    }
    strokes
}

fn wrap_angle(mut value: f64) -> f64 {
    let tau = std::f64::consts::PI * 2.0;
    while value <= -std::f64::consts::PI {
        value += tau;
    }
    while value > std::f64::consts::PI {
        value -= tau;
    }
    value
}

fn extract_features_impl(strokes: &[Vec<Point>]) -> [f64; FEATURE_COUNT] {
    let mut features = [0.0_f64; FEATURE_COUNT];
    let all_points: Vec<Point> = strokes.iter().flat_map(|stroke| stroke.iter().copied()).collect();
    if all_points.is_empty() {
        return features;
    }

    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for point in &all_points {
        x_min = x_min.min(point.x);
        x_max = x_max.max(point.x);
        y_min = y_min.min(point.y);
        y_max = y_max.max(point.y);
    }

    let x_range = (x_max - x_min).max(1e-6);
    let y_range = (y_max - y_min).max(1e-6);

    let mut cursor = 0_usize;
    features[cursor] = strokes.len() as f64;
    cursor += 1;
    features[cursor] = y_range / x_range;
    cursor += 1;
    features[cursor] = x_range / (x_range + y_range);
    cursor += 1;

    let mut angles = Vec::new();
    for stroke in strokes {
        if stroke.len() < 2 {
            continue;
        }
        for pair in stroke.windows(2) {
            let dx = pair[1].x - pair[0].x;
            let dy = pair[1].y - pair[0].y;
            angles.push(dy.atan2(dx));
        }
    }

    let mut histogram = [0.0_f64; 8];
    for angle in &angles {
        let mut bin = (((angle + std::f64::consts::PI) / (2.0 * std::f64::consts::PI)) * 8.0).floor() as i32;
        if bin < 0 {
            bin = 0;
        }
        if bin >= 8 {
            bin = 7;
        }
        histogram[bin as usize] += 1.0;
    }
    let angle_count = if angles.is_empty() { 1e-6 } else { angles.len() as f64 };
    for value in histogram {
        features[cursor] = value / angle_count;
        cursor += 1;
    }

    if angles.len() > 1 {
        let mut diffs = Vec::with_capacity(angles.len() - 1);
        for pair in angles.windows(2) {
            diffs.push(wrap_angle(pair[1] - pair[0]).abs());
        }
        let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let variance = diffs
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / diffs.len() as f64;
        features[cursor] = mean;
        cursor += 1;
        features[cursor] = variance.sqrt();
        cursor += 1;
    } else {
        cursor += 2;
    }

    let first = strokes.first().and_then(|stroke| stroke.first()).copied().unwrap_or(all_points[0]);
    let last = strokes
        .last()
        .and_then(|stroke| stroke.last())
        .copied()
        .unwrap_or(*all_points.last().unwrap());
    features[cursor] = (first.x - x_min) / x_range;
    cursor += 1;
    features[cursor] = (first.y - y_min) / y_range;
    cursor += 1;
    features[cursor] = (last.x - x_min) / x_range;
    cursor += 1;
    features[cursor] = (last.y - y_min) / y_range;
    cursor += 1;

    let mut total_length = 0.0_f64;
    let mut speeds = Vec::new();
    let mut pauses = Vec::new();
    let mut previous_end_t: Option<f64> = None;
    for stroke in strokes {
        if let Some(previous_end) = previous_end_t {
            if let Some(first_point) = stroke.first() {
                pauses.push(first_point.t - previous_end);
            }
        }
        previous_end_t = stroke.last().map(|point| point.t);

        for pair in stroke.windows(2) {
            let dx = pair[1].x - pair[0].x;
            let dy = pair[1].y - pair[0].y;
            let dt = (pair[1].t - pair[0].t).abs().max(1e-6);
            let segment = (dx * dx + dy * dy).sqrt();
            total_length += segment;
            speeds.push(segment / dt);
        }
    }
    features[cursor] = total_length / (x_range + y_range);
    cursor += 1;

    if !speeds.is_empty() {
        let mean = speeds.iter().sum::<f64>() / speeds.len() as f64;
        let variance = speeds
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / speeds.len() as f64;
        let mut sorted_speeds = speeds.clone();
        sorted_speeds.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
        let p90_index = ((sorted_speeds.len() as f64) * 0.9).floor() as usize;
        let p90 = sorted_speeds[p90_index.min(sorted_speeds.len() - 1)];
        features[cursor] = mean;
        cursor += 1;
        features[cursor] = variance.sqrt();
        cursor += 1;
        features[cursor] = p90;
        cursor += 1;
    } else {
        cursor += 3;
    }

    if !pauses.is_empty() {
        features[cursor] = pauses.iter().sum::<f64>() / pauses.len() as f64;
        cursor += 1;
        features[cursor] = pauses
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        cursor += 1;
    } else {
        cursor += 2;
    }

    for index in 0..3 {
        if index < strokes.len() && !strokes[index].is_empty() {
            let stroke = &strokes[index];
            let sx = stroke.iter().map(|point| point.x).sum::<f64>() / stroke.len() as f64;
            let sy = stroke.iter().map(|point| point.y).sum::<f64>() / stroke.len() as f64;
            features[cursor] = (sx - x_min) / x_range;
            cursor += 1;
            features[cursor] = (sy - y_min) / y_range;
            cursor += 1;
        } else {
            features[cursor] = -1.0;
            cursor += 1;
            features[cursor] = -1.0;
            cursor += 1;
        }
    }

    let mid_x = x_min + x_range * 0.5;
    let mut crossings = 0_u32;
    for stroke in strokes {
        for pair in stroke.windows(2) {
            if (pair[0].x < mid_x) != (pair[1].x < mid_x) {
                crossings += 1;
            }
        }
    }
    features[cursor] = crossings as f64;

    features
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    for index in 0..FEATURE_COUNT {
        let delta = a[index] - b[index];
        sum += delta * delta;
    }
    sum.sqrt()
}

fn label_char_to_index(label: u8) -> Option<usize> {
    if label < ALPHABET_LEN as u8 {
        Some(label as usize)
    } else {
        None
    }
}

fn top_label_from_probabilities(probabilities: &[f64]) -> Option<usize> {
    let mut best_index = None;
    let mut best_score = f64::NEG_INFINITY;
    for (index, score) in probabilities.iter().enumerate() {
        if *score > best_score {
            best_index = Some(index);
            best_score = *score;
        }
    }
    best_index
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
        let points = points_from_raw(points_ptr, point_count);
        let stroke_lengths = stroke_lengths_from_raw(stroke_lengths_ptr, stroke_count);
        let strokes = decode_strokes(points, stroke_lengths);
        extract_features_impl(&strokes)
    };

    unsafe {
        let out = slice::from_raw_parts_mut(out_ptr, FEATURE_COUNT);
        out.copy_from_slice(&features);
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
    out.fill(0.0);

    if cache_size == 0 || cache_features_ptr.is_null() || cache_labels_ptr.is_null() {
        return 0;
    }

    let cache_features = unsafe { slice::from_raw_parts(cache_features_ptr, cache_size * FEATURE_COUNT) };
    let cache_labels = unsafe { slice::from_raw_parts(cache_labels_ptr, cache_size) };

    let mut distances = Vec::with_capacity(cache_size);
    for index in 0..cache_size {
        if let Some(label_index) = label_char_to_index(cache_labels[index]) {
            let start = index * FEATURE_COUNT;
            let distance = euclidean_distance(features, &cache_features[start..start + FEATURE_COUNT]);
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
    for value in out.iter_mut() {
        if *value > 0.0 {
            *value /= total_weight;
            non_zero += 1;
        }
    }
    non_zero
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

    let sample_features = unsafe { slice::from_raw_parts(sample_features_ptr, sample_count * FEATURE_COUNT) };
    let sample_labels = unsafe { slice::from_raw_parts(sample_labels_ptr, sample_count) };
    let ready_labels = unsafe { slice::from_raw_parts(ready_labels_ptr, ready_count) };
    let out_centroids = unsafe { slice::from_raw_parts_mut(out_centroids_ptr, ready_count * FEATURE_COUNT) };
    let out_counts = unsafe { slice::from_raw_parts_mut(out_counts_ptr, ready_count) };

    out_centroids.fill(0.0);
    out_counts.fill(0);

    for (ready_index, ready_label) in ready_labels.iter().enumerate() {
        let Some(label_index) = label_char_to_index(*ready_label) else {
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

    ready_count
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

    let out = unsafe { slice::from_raw_parts_mut(out_probs_ptr, ALPHABET_LEN) };
    out.fill(0.0);

    if centroid_count == 0 {
        let uniform = 1.0 / ALPHABET_LEN as f64;
        out.fill(uniform);
        return;
    }

    let centroids = unsafe { slice::from_raw_parts(centroids_ptr, centroid_count * FEATURE_COUNT) };
    let centroid_labels = unsafe { slice::from_raw_parts(centroid_labels_ptr, centroid_count) };
    let features = unsafe { slice::from_raw_parts(features_ptr, FEATURE_COUNT) };

    let mut scores = Vec::with_capacity(centroid_count);
    let mut total = 0.0_f64;
    for index in 0..centroid_count {
        let offset = index * FEATURE_COUNT;
        let score = (-euclidean_distance(features, &centroids[offset..offset + FEATURE_COUNT])).exp();
        scores.push(score);
        total += score;
    }

    if total <= 0.0 {
        let uniform = 1.0 / centroid_count as f64;
        for label in centroid_labels {
            if let Some(label_index) = label_char_to_index(*label) {
                out[label_index] = uniform;
            }
        }
        return;
    }

    for (index, score) in scores.into_iter().enumerate() {
        if let Some(label_index) = label_char_to_index(centroid_labels[index]) {
            out[label_index] = score / total;
        }
    }
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
    out_counts.fill(0);
    out_ready.fill(0);
    out_priority.fill(0);

    if sample_count == 0 || labels_ptr.is_null() || acceptances_ptr.is_null() {
        return -1;
    }

    let labels = unsafe { slice::from_raw_parts(labels_ptr, sample_count) };
    let acceptances = unsafe { slice::from_raw_parts(acceptances_ptr, sample_count) };
    let mut totals = [0_u32; ALPHABET_LEN];
    let mut observed = 0_u32;

    for index in 0..sample_count {
        let Some(label_index) = label_char_to_index(labels[index]) else {
            continue;
        };
        let acceptance_index = if acceptances[index] == 1 { 0 } else { 1 };
        out_counts[(label_index * 2) + acceptance_index] += 1;
        totals[label_index] += 1;
        observed += 1;
    }

    let mut most_needed: i32 = -1;
    let mut min_total = u32::MAX;
    for label_index in 0..ALPHABET_LEN {
        let user_count = out_counts[label_index * 2];
        let implicit_count = out_counts[(label_index * 2) + 1];
        if user_count as usize >= MIN_READY_USER_INPUTTED_PER_LETTER
            && (user_count + implicit_count) as usize >= MIN_READY_SAMPLES_PER_LETTER
        {
            out_ready[label_index] = 1;
        }

        let total = totals[label_index];
        if observed > 0 && total < min_total {
            min_total = total;
            most_needed = label_index as i32;
        }
    }

    if observed == 0 {
        return -1;
    }

    let average = observed as f64 / ALPHABET_LEN as f64;
    for label_index in 0..ALPHABET_LEN {
        if (totals[label_index] as f64) < average * 0.85 {
            out_priority[label_index] = 1;
        }
    }

    most_needed
}

fn select_holdout_count(count: usize) -> usize {
    if count < MIN_READY_SAMPLES_PER_LETTER {
        0
    } else {
        ((count as f64 * HOLDOUT_FRACTION).floor() as usize)
            .max(1)
    }
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
    if out_training_mask_ptr.is_null() || out_holdout_mask_ptr.is_null() || out_ready_ptr.is_null() {
        return 0;
    }

    let out_training_mask = unsafe { slice::from_raw_parts_mut(out_training_mask_ptr, sample_count) };
    let out_holdout_mask = unsafe { slice::from_raw_parts_mut(out_holdout_mask_ptr, sample_count) };
    let out_ready = unsafe { slice::from_raw_parts_mut(out_ready_ptr, ALPHABET_LEN) };
    out_training_mask.fill(0);
    out_holdout_mask.fill(0);
    out_ready.fill(0);

    if sample_count == 0 || labels_ptr.is_null() || acceptances_ptr.is_null() || created_at_ptr.is_null() {
        return 0;
    }

    let labels = unsafe { slice::from_raw_parts(labels_ptr, sample_count) };
    let acceptances = unsafe { slice::from_raw_parts(acceptances_ptr, sample_count) };
    let created_at = unsafe { slice::from_raw_parts(created_at_ptr, sample_count) };

    let mut sorted_indices: Vec<usize> = (0..sample_count).collect();
    sorted_indices.sort_by(|left, right| safe_order(created_at[*left], created_at[*right]));

    let mut user_buckets: [Vec<usize>; ALPHABET_LEN] = std::array::from_fn(|_| Vec::new());
    let mut implicit_buckets: [Vec<usize>; ALPHABET_LEN] = std::array::from_fn(|_| Vec::new());

    for index in sorted_indices {
        let Some(label_index) = label_char_to_index(labels[index]) else {
            continue;
        };
        if acceptances[index] == 1 {
            user_buckets[label_index].push(index);
        } else {
            implicit_buckets[label_index].push(index);
        }
    }

    let ready_letters: Vec<usize> = (0..ALPHABET_LEN)
        .filter(|label_index| {
            user_buckets[*label_index].len() >= MIN_READY_USER_INPUTTED_PER_LETTER
                && (user_buckets[*label_index].len() + implicit_buckets[*label_index].len())
                    >= MIN_READY_SAMPLES_PER_LETTER
        })
        .collect();

    if ready_letters.len() < 2 {
        return 0;
    }

    for label_index in &ready_letters {
        out_ready[*label_index] = 1;
    }

    let per_letter_target = ready_letters
        .iter()
        .map(|label_index| user_buckets[*label_index].len() + implicit_buckets[*label_index].len())
        .min()
        .unwrap_or(0);
    if per_letter_target == 0 {
        return 0;
    }

    let target_user = ((per_letter_target as f64) * 0.2).round() as usize;
    let target_user = target_user.max(1);

    for label_index in ready_letters {
        let user_bucket = &user_buckets[label_index];
        let implicit_bucket = &implicit_buckets[label_index];
        let user_holdout_count = select_holdout_count(user_bucket.len());
        let implicit_holdout_count = select_holdout_count(implicit_bucket.len());

        for index in user_bucket.iter().skip(user_bucket.len().saturating_sub(user_holdout_count)) {
            out_holdout_mask[*index] = 1;
        }
        for index in implicit_bucket
            .iter()
            .skip(implicit_bucket.len().saturating_sub(implicit_holdout_count))
        {
            out_holdout_mask[*index] = 1;
        }

        let user_pool = &user_bucket[..user_bucket.len().saturating_sub(user_holdout_count)];
        let implicit_pool = &implicit_bucket[..implicit_bucket.len().saturating_sub(implicit_holdout_count)];

        let chosen_user_len = user_pool.len().min(target_user);
        let implicit_target = per_letter_target.saturating_sub(chosen_user_len);
        let chosen_implicit_len = implicit_pool.len().min(implicit_target);

        let mut selected = Vec::with_capacity(per_letter_target);
        selected.extend_from_slice(&user_pool[..chosen_user_len]);
        selected.extend_from_slice(&implicit_pool[..chosen_implicit_len]);

        if selected.len() < per_letter_target {
            selected.extend_from_slice(&implicit_pool[chosen_implicit_len..]);
        }
        if selected.len() < per_letter_target {
            selected.extend_from_slice(&user_pool[chosen_user_len..]);
        }

        for index in selected.into_iter().take(per_letter_target) {
            out_training_mask[index] = 1;
        }
    }

    per_letter_target as u32
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
    out_metrics.fill(0.0);

    if holdout_count == 0
        || centroid_count == 0
        || holdout_features_ptr.is_null()
        || holdout_labels_ptr.is_null()
        || holdout_acceptances_ptr.is_null()
        || centroids_ptr.is_null()
        || centroid_labels_ptr.is_null()
    {
        return;
    }

    let holdout_features = unsafe { slice::from_raw_parts(holdout_features_ptr, holdout_count * FEATURE_COUNT) };
    let holdout_labels = unsafe { slice::from_raw_parts(holdout_labels_ptr, holdout_count) };
    let holdout_acceptances = unsafe { slice::from_raw_parts(holdout_acceptances_ptr, holdout_count) };
    let centroids = unsafe { slice::from_raw_parts(centroids_ptr, centroid_count * FEATURE_COUNT) };
    let centroid_labels = unsafe { slice::from_raw_parts(centroid_labels_ptr, centroid_count) };

    let mut user_total = 0_u32;
    let mut user_correct = 0_u32;
    let mut implicit_total = 0_u32;
    let mut implicit_correct = 0_u32;
    let mut overall_total = 0_u32;
    let mut overall_correct = 0_u32;
    let mut probabilities = [0.0_f64; ALPHABET_LEN];

    for sample_index in 0..holdout_count {
        probabilities.fill(0.0);
        let feature_offset = sample_index * FEATURE_COUNT;
        let features = &holdout_features[feature_offset..feature_offset + FEATURE_COUNT];

        let mut scores = Vec::with_capacity(centroid_count);
        let mut total = 0.0_f64;
        for centroid_index in 0..centroid_count {
            let centroid_offset = centroid_index * FEATURE_COUNT;
            let score = (-euclidean_distance(features, &centroids[centroid_offset..centroid_offset + FEATURE_COUNT]))
                .exp();
            scores.push(score);
            total += score;
        }

        if total > 0.0 {
            for (centroid_index, score) in scores.into_iter().enumerate() {
                if let Some(label_index) = label_char_to_index(centroid_labels[centroid_index]) {
                    probabilities[label_index] = score / total;
                }
            }
        }

        let predicted = top_label_from_probabilities(&probabilities);
        let actual = label_char_to_index(holdout_labels[sample_index]);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f64, y: f64, t: f64) -> Point {
        Point { x, y, t }
    }

    fn feature_vec(value: f64) -> [f64; FEATURE_COUNT] {
        [value; FEATURE_COUNT]
    }

    fn flatten_features(features: &[[f64; FEATURE_COUNT]]) -> Vec<f64> {
        features.iter().flat_map(|item| item.iter().copied()).collect()
    }

    #[test]
    fn extract_features_is_deterministic_and_finite() {
        let strokes = vec![
            vec![point(0.0, 0.0, 0.0), point(1.0, 1.0, 1.0), point(2.0, 1.0, 2.0)],
            vec![point(2.0, 1.0, 3.0), point(1.0, 0.0, 4.0)],
        ];

        let first = extract_features_impl(&strokes);
        let second = extract_features_impl(&strokes);

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
    fn knn_predict_respects_far_neighbor_guard() {
        let query = feature_vec(0.0);
        let cache = flatten_features(&[feature_vec(100.0)]);
        let labels = [0_u8];
        let mut out = [0.0_f64; ALPHABET_LEN];

        let non_zero = knn_predict(
            query.as_ptr(),
            cache.as_ptr(),
            labels.as_ptr(),
            labels.len(),
            1,
            1.0,
            out.as_mut_ptr(),
        );

        assert_eq!(non_zero, 0);
        assert!(out.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn centroid_training_and_prediction_match_expected_label() {
        let samples = flatten_features(&[
            feature_vec(1.0),
            feature_vec(1.2),
            feature_vec(9.0),
            feature_vec(9.3),
        ]);
        let sample_labels = [2_u8, 2_u8, 4_u8, 4_u8];
        let ready_labels = [2_u8, 4_u8];
        let mut centroids = [0.0_f64; FEATURE_COUNT * 2];
        let mut counts = [0_u32; 2];

        let trained = train_centroid_classifier(
            samples.as_ptr(),
            sample_labels.as_ptr(),
            sample_labels.len(),
            ready_labels.as_ptr(),
            ready_labels.len(),
            centroids.as_mut_ptr(),
            counts.as_mut_ptr(),
        );

        assert_eq!(trained, 2);
        assert_eq!(counts, [2, 2]);

        let query = feature_vec(9.1);
        let mut probs = [0.0_f64; ALPHABET_LEN];
        predict_centroid_classifier(
            centroids.as_ptr(),
            ready_labels.as_ptr(),
            ready_labels.len(),
            query.as_ptr(),
            probs.as_mut_ptr(),
        );

        assert!(probs[4] > probs[2]);
    }

    #[test]
    fn compute_letter_stats_sets_ready_priority_and_most_needed() {
        let labels = [0_u8, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        let acceptances = [1_u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut counts = [0_u32; ALPHABET_LEN * 2];
        let mut ready = [0_u8; ALPHABET_LEN];
        let mut priority = [0_u8; ALPHABET_LEN];

        let most_needed = compute_letter_stats(
            labels.as_ptr(),
            acceptances.as_ptr(),
            labels.len(),
            counts.as_mut_ptr(),
            ready.as_mut_ptr(),
            priority.as_mut_ptr(),
        );

        assert_eq!(counts[0], 1);
        assert_eq!(counts[1], 4);
        assert_eq!(ready[0], 1);
        assert_eq!(ready[1], 0);
        assert_eq!(most_needed, 2);
        assert_eq!(priority[0], 0);
        assert_eq!(priority[2], 1);
    }

    #[test]
    fn build_balanced_dataset_marks_ready_letters_and_masks() {
        let labels = [0_u8, 0, 0, 0, 0, 1, 1, 1, 1, 1];
        let acceptances = [1_u8, 0, 0, 0, 0, 1, 0, 0, 0, 0];
        let created_at = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut training_mask = [0_u8; 10];
        let mut holdout_mask = [0_u8; 10];
        let mut ready = [0_u8; ALPHABET_LEN];

        let per_letter_target = build_balanced_dataset(
            labels.as_ptr(),
            acceptances.as_ptr(),
            created_at.as_ptr(),
            labels.len(),
            training_mask.as_mut_ptr(),
            holdout_mask.as_mut_ptr(),
            ready.as_mut_ptr(),
        );

        assert_eq!(per_letter_target, 5);
        assert_eq!(ready[0], 1);
        assert_eq!(ready[1], 1);
        assert!(training_mask.iter().all(|value| *value == 1));
        assert!(holdout_mask.iter().all(|value| *value == 0));
    }

    #[test]
    fn evaluate_snapshot_reports_split_and_overall_accuracy() {
        let holdout = flatten_features(&[
            feature_vec(0.0),
            feature_vec(10.0),
            feature_vec(9.5),
            feature_vec(0.1),
        ]);
        let holdout_labels = [0_u8, 1_u8, 0_u8, 0_u8];
        let holdout_acceptances = [1_u8, 0_u8, 1_u8, 0_u8];
        let centroids = flatten_features(&[feature_vec(0.0), feature_vec(10.0)]);
        let centroid_labels = [0_u8, 1_u8];
        let mut metrics = [0.0_f64; 3];

        evaluate_snapshot(
            holdout.as_ptr(),
            holdout_labels.as_ptr(),
            holdout_acceptances.as_ptr(),
            holdout_labels.len(),
            centroids.as_ptr(),
            centroid_labels.as_ptr(),
            centroid_labels.len(),
            metrics.as_mut_ptr(),
        );

        assert!((metrics[0] - 0.5).abs() < 1e-9);
        assert!((metrics[1] - 1.0).abs() < 1e-9);
        assert!((metrics[2] - 0.75).abs() < 1e-9);
    }
}
