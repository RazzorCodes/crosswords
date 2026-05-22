use std::path::Path;

use serde::Deserialize;

use crate::logger::log_progress;
#[cfg(test)]
use crate::logger::format_log_line;
use super::bridge::batcher::DualSample;
use super::dataset::{branch_sample, load_annotated_dataset, BranchSample, RawPoint, RawSample};
use super::preproc::raster::rasterize;
use super::preproc::sequence::to_1d_cnn;
use super::preproc::spatial::normalize_unit_box;
use super::preproc::svm::{flatten_features, total_curvature};
use super::preproc::temporal::resample_default;
use super::preproc::{
    StrokePath, StrokePoint, CNN_1D_CHANNELS, RASTER_SIZE, RESAMPLED_POINTS, SVM_FEATURES,
};

#[derive(Clone, Debug, Default)]
pub struct PipelineOutput {
    pub one_d_cnn: Vec<BranchSample>,
    pub two_d_cnn: Vec<BranchSample>,
    pub svm: Vec<BranchSample>,
    pub skipped: usize,
}

fn regroup_by_stroke(points: &[StrokePoint]) -> StrokePath {
    let mut strokes = StrokePath::new();
    for point in points {
        if strokes.len() <= point.stroke_index {
            strokes.resize_with(point.stroke_index + 1, Vec::new);
        }
        strokes[point.stroke_index].push(point.clone());
    }
    strokes
        .into_iter()
        .filter(|stroke| !stroke.is_empty())
        .collect()
}

pub(crate) fn preprocess_sample_for_cnn(sample: &RawSample) -> Result<DualSample, String> {
    let label_index = sample
        .label_index()
        .ok_or_else(|| format!("sample {} has invalid label", sample.sample_id))?;
    let clean = sample
        .clean()
        .ok_or_else(|| format!("sample {} has no usable strokes", sample.sample_id))?;
    let raw_point_count = clean.strokes.iter().map(Vec::len).sum::<usize>();
    if raw_point_count < 2 {
        return Err(format!(
            "sample {} has fewer than 2 raw stroke points",
            sample.sample_id
        ));
    }
    let (normalized, _) = normalize_unit_box(&clean.strokes)
        .ok_or_else(|| format!("sample {} could not be normalized", sample.sample_id))?;
    let resampled = resample_default(&normalized);
    if resampled.len() != RESAMPLED_POINTS {
        return Err(format!("sample {} failed resampling", sample.sample_id));
    }
    let resampled_strokes = regroup_by_stroke(&resampled);
    Ok(DualSample {
        sample_id: sample.sample_id.clone(),
        label_index,
        one_d_values: to_1d_cnn(&resampled),
        two_d_values: rasterize(&resampled_strokes),
    })
}

#[derive(Debug, Deserialize)]
struct StrokesInput {
    strokes: Vec<Vec<RawPoint>>,
}

pub fn preprocess_strokes_for_cnn(input_json: &str) -> Result<DualSample, String> {
    let input: StrokesInput = serde_json::from_str(input_json)
        .map_err(|error| format!("failed to parse strokes input: {error}"))?;
    let sample = RawSample {
        sample_id: "input".to_string(),
        sample_label: "A".to_string(),
        extra_labels: vec![],
        strokes: input.strokes,
    };
    let mut output = preprocess_sample_for_cnn(&sample)?;
    output.label_index = 0;
    Ok(output)
}

pub fn preprocess_file(source: &Path) -> Result<PipelineOutput, String> {
    log_progress("preproc.pipeline", 0, Some(source), "loading dataset");
    let dataset = load_annotated_dataset(source)?;
    let total = dataset.samples.len();
    log_progress(
        "preproc.pipeline",
        1,
        Some(source),
        &format!("loaded {} samples from {}", total, dataset.name),
    );

    let mut output = PipelineOutput::default();
    if total == 0 {
        return Err(format!("{} contains no samples", source.display()));
    }

    for (index, sample) in dataset.samples.iter().enumerate() {
        let percent = 1 + (((index + 1) as f64 / total as f64) * 98.0).floor() as u8;

        let Some(label_index) = sample.label_index() else {
            output.skipped += 1;
            log_progress(
                "preproc.pipeline",
                percent,
                Some(source),
                &format!("skipped {}: invalid label", sample.sample_id),
            );
            continue;
        };
        let Some(clean) = sample.clean() else {
            output.skipped += 1;
            log_progress(
                "preproc.pipeline",
                percent,
                Some(source),
                &format!("skipped {}: no usable strokes", sample.sample_id),
            );
            continue;
        };
        let raw_point_count = clean.strokes.iter().map(Vec::len).sum::<usize>();
        if raw_point_count < 2 {
            output.skipped += 1;
            log_progress(
                "preproc.pipeline",
                percent,
                Some(source),
                &format!(
                    "skipped {}: fewer than 2 raw stroke points",
                    sample.sample_id
                ),
            );
            continue;
        }
        let Some((normalized, bounds)) = normalize_unit_box(&clean.strokes) else {
            output.skipped += 1;
            continue;
        };

        let resampled = resample_default(&normalized);
        if resampled.len() != RESAMPLED_POINTS {
            output.skipped += 1;
            log_progress(
                "preproc.pipeline",
                percent,
                Some(source),
                &format!("skipped {}: failed resampling", sample.sample_id),
            );
            continue;
        }
        let resampled_strokes = regroup_by_stroke(&resampled);
        let curvature = total_curvature(&resampled_strokes);

        let mut one_d = branch_sample(
            sample,
            vec![CNN_1D_CHANNELS, RESAMPLED_POINTS],
            to_1d_cnn(&resampled),
        );
        one_d.label_index = label_index;
        one_d.raw_point_count = raw_point_count;
        output.one_d_cnn.push(one_d);

        let mut two_d = branch_sample(
            sample,
            vec![1, RASTER_SIZE, RASTER_SIZE],
            rasterize(&resampled_strokes),
        );
        two_d.label_index = label_index;
        two_d.raw_point_count = raw_point_count;
        output.two_d_cnn.push(two_d);

        let mut svm = branch_sample(
            sample,
            vec![SVM_FEATURES],
            flatten_features(&resampled, bounds, clean.strokes.len(), curvature),
        );
        svm.label_index = label_index;
        svm.raw_point_count = raw_point_count;
        output.svm.push(svm);

        if index == total - 1 || (index + 1) % 25 == 0 {
            log_progress(
                "preproc.pipeline",
                percent,
                Some(source),
                &format!("processed {}/{} samples", index + 1, total),
            );
        }
    }

    if output.one_d_cnn.is_empty() {
        return Err(format!("{} produced no valid samples", source.display()));
    }

    log_progress(
        "preproc.pipeline",
        99,
        Some(source),
        &format!(
            "prepared {} valid samples; skipped {}",
            output.one_d_cnn.len(),
            output.skipped
        ),
    );

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn format_log_line_includes_optional_file() {
        let line = format_log_line(
            "preproc.pipeline",
            42,
            Some(Path::new("data/annotated/regular-0001.json")),
            "processed 1/2 samples",
        );
        assert!(line.contains("preproc.pipeline [042%] data/annotated/regular-0001.json"));
    }

    #[test]
    fn pipeline_preprocesses_small_fixture() {
        let path = std::env::temp_dir().join(format!(
            "crosswords-preproc-test-{}.json",
            std::process::id()
        ));
        fs::write(
            &path,
            r#"{
              "fixture": [
                {
                  "sample_id": "s1",
                  "sample_label": "A",
                  "extra_labels": ["test"],
                  "strokes": [[
                    {"x": 0.0, "y": 0.0, "t": 0.0},
                    {"x": 10.0, "y": 0.0, "t": 1.0}
                  ]]
                }
              ]
            }"#,
        )
        .unwrap();

        let output = preprocess_file(&path).unwrap();
        let _ = fs::remove_file(&path);

        assert_eq!(output.one_d_cnn.len(), 1);
        assert_eq!(output.one_d_cnn[0].shape, vec![3, 64]);
        assert_eq!(output.two_d_cnn[0].shape, vec![1, 28, 28]);
        assert_eq!(output.svm[0].shape, vec![131]);
    }

    #[test]
    fn strokes_inference_preprocess_uses_resampled_cnn_shapes() {
        let sample = preprocess_strokes_for_cnn(
            r#"{"strokes":[[{"x":0.0,"y":0.0,"t":0.0},{"x":10.0,"y":0.0,"t":1.0}]]}"#,
        )
        .unwrap();
        assert_eq!(sample.one_d_values.len(), 3 * 64);
        assert_eq!(sample.two_d_values.len(), 28 * 28);
    }
}
