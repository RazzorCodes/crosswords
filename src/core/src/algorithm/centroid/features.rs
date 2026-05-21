use crate::types::{Point, FEATURE_COUNT};
use std::cmp::Ordering;

pub fn decode_strokes(points: &[f64], stroke_lengths: &[u32]) -> Vec<Vec<Point>> {
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

pub fn extract_features(strokes: &[Vec<Point>]) -> [f64; FEATURE_COUNT] {
    let mut features = [0.0_f64; FEATURE_COUNT];
    let all_points: Vec<Point> = strokes
        .iter()
        .flat_map(|stroke| stroke.iter().copied())
        .collect();
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
            angles.push((pair[1].y - pair[0].y).atan2(pair[1].x - pair[0].x));
        }
    }

    let mut histogram = [0.0_f64; 8];
    for angle in &angles {
        let mut bin =
            (((angle + std::f64::consts::PI) / (2.0 * std::f64::consts::PI)) * 8.0).floor() as i32;
        if bin < 0 {
            bin = 0;
        }
        if bin >= 8 {
            bin = 7;
        }
        histogram[bin as usize] += 1.0;
    }
    let angle_count = if angles.is_empty() {
        1e-6
    } else {
        angles.len() as f64
    };
    for value in histogram {
        features[cursor] = value / angle_count;
        cursor += 1;
    }

    if angles.len() > 1 {
        let diffs: Vec<f64> = angles
            .windows(2)
            .map(|pair| wrap_angle(pair[1] - pair[0]).abs())
            .collect();
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

    let first = strokes
        .first()
        .and_then(|stroke| stroke.first())
        .copied()
        .unwrap_or(all_points[0]);
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
        features[cursor] = mean;
        cursor += 1;
        features[cursor] = variance.sqrt();
        cursor += 1;
        features[cursor] = sorted_speeds[p90_index.min(sorted_speeds.len() - 1)];
        cursor += 1;
    } else {
        cursor += 3;
    }

    if !pauses.is_empty() {
        features[cursor] = pauses.iter().sum::<f64>() / pauses.len() as f64;
        cursor += 1;
        features[cursor] = pauses.iter().copied().fold(f64::NEG_INFINITY, f64::max);
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
