use crate::data::dataset::RawPoint;

use super::{StrokePath, StrokePoint};

const EPSILON: f64 = 1e-9;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
}

impl Bounds {
    pub fn width(self) -> f64 {
        self.max_x - self.min_x
    }

    pub fn height(self) -> f64 {
        self.max_y - self.min_y
    }

    pub fn aspect_ratio(self) -> f64 {
        self.width() / self.height().max(EPSILON)
    }
}

pub fn bounds_for_raw(strokes: &[Vec<RawPoint>]) -> Option<Bounds> {
    let mut bounds = Bounds {
        min_x: f64::INFINITY,
        max_x: f64::NEG_INFINITY,
        min_y: f64::INFINITY,
        max_y: f64::NEG_INFINITY,
    };
    let mut seen = false;

    for point in strokes.iter().flat_map(|stroke| stroke.iter()) {
        bounds.min_x = bounds.min_x.min(point.x);
        bounds.max_x = bounds.max_x.max(point.x);
        bounds.min_y = bounds.min_y.min(point.y);
        bounds.max_y = bounds.max_y.max(point.y);
        seen = true;
    }

    seen.then_some(bounds)
}

pub fn normalize_unit_box(strokes: &[Vec<RawPoint>]) -> Option<(StrokePath, Bounds)> {
    let bounds = bounds_for_raw(strokes)?;
    let width = bounds.width();
    let height = bounds.height();
    let scale = width.max(height).max(EPSILON);
    let normalized_width = width / scale;
    let normalized_height = height / scale;
    let x_offset = (1.0 - normalized_width) * 0.5;
    let y_offset = (1.0 - normalized_height) * 0.5;

    let normalized = strokes
        .iter()
        .enumerate()
        .map(|(stroke_index, stroke)| {
            stroke
                .iter()
                .map(|point| StrokePoint {
                    x: ((point.x - bounds.min_x) / scale) + x_offset,
                    y: ((point.y - bounds.min_y) / scale) + y_offset,
                    t: point.t,
                    stroke_index,
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    Some((normalized, bounds))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f64, y: f64) -> RawPoint {
        RawPoint { x, y, t: 0.0 }
    }

    #[test]
    fn normalize_preserves_aspect_ratio_and_centers() {
        let strokes = vec![vec![point(10.0, 5.0), point(30.0, 15.0)]];
        let (normalized, bounds) = normalize_unit_box(&strokes).unwrap();
        let xs = normalized[0]
            .iter()
            .map(|point| point.x)
            .collect::<Vec<_>>();
        let ys = normalized[0]
            .iter()
            .map(|point| point.y)
            .collect::<Vec<_>>();

        assert_eq!(bounds.aspect_ratio(), 2.0);
        assert!((xs[0] - 0.0).abs() < 1e-9);
        assert!((xs[1] - 1.0).abs() < 1e-9);
        assert!((ys[0] - 0.25).abs() < 1e-9);
        assert!((ys[1] - 0.75).abs() < 1e-9);
    }

    #[test]
    fn normalize_degenerate_shape_is_finite() {
        let strokes = vec![vec![point(3.0, 4.0)]];
        let (normalized, _) = normalize_unit_box(&strokes).unwrap();
        assert!(normalized[0][0].x.is_finite());
        assert!(normalized[0][0].y.is_finite());
        assert!((normalized[0][0].x - 0.5).abs() < 1e-9);
        assert!((normalized[0][0].y - 0.5).abs() < 1e-9);
    }
}
