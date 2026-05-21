pub const ALPHABET_LEN: usize = 26;
pub const FEATURE_COUNT: usize = 30;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub t: f64,
}

pub fn label_to_index(label: u8) -> Option<usize> {
    match label {
        0..=25 => Some(label as usize),
        b'A'..=b'Z' => Some((label - b'A') as usize),
        b'a'..=b'z' => Some((label - b'a') as usize),
        _ => None,
    }
}

pub fn label_str_to_index(label: &str) -> Option<usize> {
    label
        .as_bytes()
        .first()
        .and_then(|value| label_to_index(*value))
}

pub fn euclidean_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| {
            let delta = a - b;
            delta * delta
        })
        .sum::<f64>()
        .sqrt()
}

pub fn top_label_from_probabilities(probabilities: &[f64]) -> Option<usize> {
    probabilities
        .iter()
        .take(ALPHABET_LEN)
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
}
