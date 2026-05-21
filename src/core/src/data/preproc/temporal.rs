use super::{StrokePath, StrokePoint, RESAMPLED_POINTS};

#[derive(Clone, Debug)]
struct Segment {
    start: StrokePoint,
    end: StrokePoint,
    start_distance: f64,
    end_distance: f64,
}

fn distance(left: &StrokePoint, right: &StrokePoint) -> f64 {
    let dx = right.x - left.x;
    let dy = right.y - left.y;
    (dx * dx + dy * dy).sqrt()
}

pub fn resample(strokes: &StrokePath, count: usize) -> Vec<StrokePoint> {
    if count == 0 {
        return Vec::new();
    }

    let first_point = strokes.iter().find_map(|stroke| stroke.first()).cloned();
    let Some(first_point) = first_point else {
        return Vec::new();
    };

    let mut segments = Vec::new();
    let mut total = 0.0;
    for stroke in strokes {
        for pair in stroke.windows(2) {
            let length = distance(&pair[0], &pair[1]);
            if length <= 1e-12 {
                continue;
            }
            segments.push(Segment {
                start: pair[0].clone(),
                end: pair[1].clone(),
                start_distance: total,
                end_distance: total + length,
            });
            total += length;
        }
    }

    if segments.is_empty() || total <= 1e-12 {
        return (0..count).map(|_| first_point.clone()).collect();
    }

    let last_point = segments.last().map(|segment| segment.end.clone()).unwrap();
    let step = if count == 1 {
        0.0
    } else {
        total / (count - 1) as f64
    };
    let mut segment_index = 0_usize;
    let mut result = Vec::with_capacity(count);

    for index in 0..count {
        if index == 0 {
            result.push(first_point.clone());
            continue;
        }
        if index == count - 1 {
            result.push(last_point.clone());
            continue;
        }

        let target = step * index as f64;
        while segment_index + 1 < segments.len() && segments[segment_index].end_distance < target {
            segment_index += 1;
        }
        let segment = &segments[segment_index];
        let local = (target - segment.start_distance)
            / (segment.end_distance - segment.start_distance).max(1e-12);
        result.push(StrokePoint {
            x: segment.start.x + (segment.end.x - segment.start.x) * local,
            y: segment.start.y + (segment.end.y - segment.start.y) * local,
            t: segment.start.t + (segment.end.t - segment.start.t) * local,
            stroke_index: segment.start.stroke_index,
        });
    }

    result
}

pub fn resample_default(strokes: &StrokePath) -> Vec<StrokePoint> {
    resample(strokes, RESAMPLED_POINTS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f64, stroke_index: usize) -> StrokePoint {
        StrokePoint {
            x,
            y: 0.0,
            t: x,
            stroke_index,
        }
    }

    #[test]
    fn resample_returns_requested_count() {
        let strokes = vec![vec![point(0.0, 0), point(10.0, 0)]];
        let points = resample(&strokes, 64);
        assert_eq!(points.len(), 64);
        assert!((points[0].x - 0.0).abs() < 1e-9);
        assert!((points[63].x - 10.0).abs() < 1e-9);
    }

    #[test]
    fn resample_preserves_stroke_breaks() {
        let strokes = vec![
            vec![point(0.0, 0), point(1.0, 0)],
            vec![point(10.0, 1), point(11.0, 1)],
        ];
        let points = resample(&strokes, 5);
        assert!(points.iter().any(|point| point.stroke_index == 0));
        assert!(points.iter().any(|point| point.stroke_index == 1));
        assert!(!points.iter().any(|point| point.x > 1.0 && point.x < 10.0));
    }

    #[test]
    fn resample_handles_single_point_stroke() {
        let strokes = vec![vec![point(4.0, 0)]];
        let points = resample(&strokes, 64);
        assert_eq!(points.len(), 64);
        assert!(points.iter().all(|point| point.x == 4.0));
    }
}
