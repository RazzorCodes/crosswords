use super::{StrokePoint, CNN_1D_CHANNELS, RESAMPLED_POINTS};

pub fn to_1d_cnn(points: &[StrokePoint]) -> Vec<f64> {
    let mut values = vec![0.0; CNN_1D_CHANNELS * RESAMPLED_POINTS];
    for index in 0..RESAMPLED_POINTS.min(points.len()) {
        let is_new_stroke = index == 0
            || points[index].stroke_index != points[index.saturating_sub(1)].stroke_index;
        values[index] = if is_new_stroke {
            0.0
        } else {
            points[index].x - points[index - 1].x
        };
        values[RESAMPLED_POINTS + index] = if is_new_stroke {
            0.0
        } else {
            points[index].y - points[index - 1].y
        };
        values[RESAMPLED_POINTS * 2 + index] = if is_new_stroke { 1.0 } else { 0.0 };
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f64, stroke_index: usize) -> StrokePoint {
        StrokePoint {
            x,
            y: 0.0,
            t: 0.0,
            stroke_index,
        }
    }

    #[test]
    fn sequence_has_expected_shape_and_pen_state() {
        let mut points = vec![point(0.0, 0), point(0.5, 0), point(0.8, 1)];
        points.resize(RESAMPLED_POINTS, point(1.0, 1));

        let values = to_1d_cnn(&points);
        assert_eq!(values.len(), 3 * 64);
        assert_eq!(values[0], 0.0);
        assert_eq!(values[1], 0.5);
        assert_eq!(values[2], 0.0);
        assert_eq!(values[128], 1.0);
        assert_eq!(values[129], 0.0);
        assert_eq!(values[130], 1.0);
    }
}
