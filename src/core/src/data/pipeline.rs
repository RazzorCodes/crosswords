use chrono::Utc;
use std::path::Path;

use super::dataset::{branch_sample, load_annotated_dataset, BranchSample};
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

fn timestamp() -> String {
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

pub fn format_log_line(component: &str, percent: u8, file: Option<&Path>, message: &str) -> String {
    match file {
        Some(file) => format!(
            "[{}] {} [{:03}%] {} {}",
            timestamp(),
            component,
            percent.min(100),
            file.display(),
            message
        ),
        None => format!(
            "[{}] {} [{:03}%] {}",
            timestamp(),
            component,
            percent.min(100),
            message
        ),
    }
}

pub fn log_progress(component: &str, percent: u8, file: Option<&Path>, message: &str) {
    eprintln!("{}", format_log_line(component, percent, file, message));
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
        output.one_d_cnn.push(one_d);

        let mut two_d = branch_sample(
            sample,
            vec![1, RASTER_SIZE, RASTER_SIZE],
            rasterize(&resampled_strokes),
        );
        two_d.label_index = label_index;
        output.two_d_cnn.push(two_d);

        let mut svm = branch_sample(
            sample,
            vec![SVM_FEATURES],
            flatten_features(&resampled, bounds, clean.strokes.len(), curvature),
        );
        svm.label_index = label_index;
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
}
