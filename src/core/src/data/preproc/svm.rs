use super::{StrokePath, StrokePoint, RESAMPLED_POINTS, SVM_FEATURES};
use crate::data::preproc::spatial::Bounds;

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

pub fn total_curvature(strokes: &StrokePath) -> f64 {
    let mut total = 0.0;
    for stroke in strokes {
        let angles = stroke
            .windows(2)
            .map(|pair| (pair[1].y - pair[0].y).atan2(pair[1].x - pair[0].x))
            .collect::<Vec<_>>();
        for pair in angles.windows(2) {
            total += wrap_angle(pair[1] - pair[0]).abs();
        }
    }
    total
}

pub fn flatten_features(
    points: &[StrokePoint],
    bounds: Bounds,
    stroke_count: usize,
    curvature: f64,
) -> Vec<f64> {
    let mut values = Vec::with_capacity(SVM_FEATURES);
    for index in 0..RESAMPLED_POINTS {
        if let Some(point) = points.get(index) {
            values.push(point.x);
            values.push(point.y);
        } else {
            values.push(0.0);
            values.push(0.0);
        }
    }
    values.push(bounds.aspect_ratio());
    values.push(stroke_count as f64);
    values.push(curvature);
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f64, y: f64) -> StrokePoint {
        StrokePoint {
            x,
            y,
            t: 0.0,
            stroke_index: 0,
        }
    }

    #[test]
    fn svm_features_have_expected_length() {
        let points = vec![point(0.0, 0.0); RESAMPLED_POINTS];
        let bounds = Bounds {
            min_x: 0.0,
            max_x: 2.0,
            min_y: 0.0,
            max_y: 1.0,
        };
        let values = flatten_features(&points, bounds, 2, 1.5);
        assert_eq!(values.len(), 131);
        assert_eq!(values[128], 2.0);
        assert_eq!(values[129], 2.0);
        assert_eq!(values[130], 1.5);
    }

    #[test]
    fn curvature_sums_angle_changes_within_strokes() {
        let strokes = vec![vec![point(0.0, 0.0), point(1.0, 0.0), point(1.0, 1.0)]];
        let curvature = total_curvature(&strokes);
        assert!((curvature - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    }
}
