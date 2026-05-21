use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::types::label_str_to_index;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RawPoint {
    pub x: f64,
    pub y: f64,
    pub t: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RawSample {
    pub sample_id: String,
    pub sample_label: String,
    #[serde(default)]
    pub extra_labels: Vec<String>,
    pub strokes: Vec<Vec<RawPoint>>,
}

#[derive(Clone, Debug)]
pub struct Dataset {
    pub name: String,
    pub samples: Vec<RawSample>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BranchSample {
    pub sample_id: String,
    pub sample_label: String,
    pub label_index: usize,
    pub extra_labels: Vec<String>,
    pub shape: Vec<usize>,
    pub values: Vec<f64>,
    #[serde(default)]
    pub raw_point_count: usize,
}

#[derive(Clone, Debug, Default)]
pub struct CleanSample {
    pub strokes: Vec<Vec<RawPoint>>,
}

impl RawSample {
    pub fn label_index(&self) -> Option<usize> {
        label_str_to_index(&self.sample_label)
    }

    pub fn clean(&self) -> Option<CleanSample> {
        let strokes = self
            .strokes
            .iter()
            .filter_map(|stroke| {
                let points = stroke
                    .iter()
                    .filter(|point| {
                        point.x.is_finite() && point.y.is_finite() && point.t.is_finite()
                    })
                    .cloned()
                    .collect::<Vec<_>>();
                (!points.is_empty()).then_some(points)
            })
            .collect::<Vec<_>>();

        (!strokes.is_empty()).then_some(CleanSample { strokes })
    }
}

pub fn load_annotated_dataset(path: &Path) -> Result<Dataset, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let mut payload: BTreeMap<String, Vec<RawSample>> = serde_json::from_str(&text)
        .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
    if payload.len() != 1 {
        return Err(format!(
            "expected one top-level dataset key in {}, found {}",
            path.display(),
            payload.len()
        ));
    }
    let (name, samples) = payload.pop_first().unwrap();
    Ok(Dataset { name, samples })
}

pub fn branch_sample(sample: &RawSample, shape: Vec<usize>, values: Vec<f64>) -> BranchSample {
    BranchSample {
        sample_id: sample.sample_id.clone(),
        sample_label: sample.sample_label.clone(),
        label_index: sample.label_index().unwrap_or(usize::MAX),
        extra_labels: sample.extra_labels.clone(),
        shape,
        values,
        raw_point_count: sample.strokes.iter().map(Vec::len).sum(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_drops_empty_and_non_finite_points() {
        let sample = RawSample {
            sample_id: "s1".to_string(),
            sample_label: "A".to_string(),
            extra_labels: vec![],
            strokes: vec![
                vec![],
                vec![
                    RawPoint {
                        x: 0.0,
                        y: 1.0,
                        t: 0.0,
                    },
                    RawPoint {
                        x: f64::NAN,
                        y: 1.0,
                        t: 0.0,
                    },
                ],
            ],
        };

        let clean = sample.clean().unwrap();
        assert_eq!(clean.strokes.len(), 1);
        assert_eq!(clean.strokes[0].len(), 1);
    }
}
