use std::cmp::Ordering;

pub const ALPHABET_LEN: usize = 26;
pub const FEATURE_COUNT: usize = 30;
pub const MIN_READY_SAMPLES_PER_LETTER: usize = 5;
pub const MIN_READY_USER_INPUTTED_PER_LETTER: usize = 1;
pub const HOLDOUT_FRACTION: f64 = 0.2;

#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub t: f64,
}

pub fn safe_desc_order(left: f64, right: f64) -> Ordering {
    right.partial_cmp(&left).unwrap_or(Ordering::Equal)
}

pub fn label_to_index(label: u8) -> Option<usize> {
    if label < ALPHABET_LEN as u8 {
        Some(label as usize)
    } else {
        None
    }
}

pub fn top_label_from_probabilities(probabilities: &[f64]) -> Option<usize> {
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

pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    for index in 0..FEATURE_COUNT {
        let delta = a[index] - b[index];
        sum += delta * delta;
    }
    sum.sqrt()
}
