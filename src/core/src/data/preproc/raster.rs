use super::{StrokePath, RASTER_SIZE};

const STROKE_RADIUS: f64 = 1.0;
const AA_WIDTH: f64 = 1.0;

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn to_pixel(value: f64) -> f64 {
    clamp01(value) * (RASTER_SIZE as f64 - 1.0)
}

fn distance_to_segment(px: f64, py: f64, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let vx = bx - ax;
    let vy = by - ay;
    let wx = px - ax;
    let wy = py - ay;
    let len_sq = vx * vx + vy * vy;
    if len_sq <= 1e-12 {
        let dx = px - ax;
        let dy = py - ay;
        return (dx * dx + dy * dy).sqrt();
    }
    let projection = ((wx * vx + wy * vy) / len_sq).clamp(0.0, 1.0);
    let cx = ax + projection * vx;
    let cy = ay + projection * vy;
    let dx = px - cx;
    let dy = py - cy;
    (dx * dx + dy * dy).sqrt()
}

fn paint_segment(values: &mut [f64], ax: f64, ay: f64, bx: f64, by: f64) {
    let min_x = ax.min(bx).floor().max(0.0) as usize;
    let max_x = ax.max(bx).ceil().min(RASTER_SIZE as f64 - 1.0) as usize;
    let min_y = ay.min(by).floor().max(0.0) as usize;
    let max_y = ay.max(by).ceil().min(RASTER_SIZE as f64 - 1.0) as usize;

    let margin = (STROKE_RADIUS + AA_WIDTH).ceil() as usize;
    let start_y = min_y.saturating_sub(margin);
    let end_y = (max_y + margin).min(RASTER_SIZE - 1);
    let start_x = min_x.saturating_sub(margin);
    let end_x = (max_x + margin).min(RASTER_SIZE - 1);

    for y in start_y..=end_y {
        for x in start_x..=end_x {
            let distance = distance_to_segment(x as f64, y as f64, ax, ay, bx, by);
            let coverage = if distance <= STROKE_RADIUS {
                1.0
            } else if distance <= STROKE_RADIUS + AA_WIDTH {
                1.0 - ((distance - STROKE_RADIUS) / AA_WIDTH)
            } else {
                0.0
            };
            let offset = y * RASTER_SIZE + x;
            values[offset] = values[offset].max(coverage);
        }
    }
}

pub fn rasterize(strokes: &StrokePath) -> Vec<f64> {
    let mut values = vec![0.0_f64; RASTER_SIZE * RASTER_SIZE];

    for stroke in strokes {
        if stroke.is_empty() {
            continue;
        }
        if stroke.len() == 1 {
            let px = to_pixel(stroke[0].x);
            let py = to_pixel(stroke[0].y);
            paint_segment(&mut values, px, py, px, py);
            continue;
        }
        for pair in stroke.windows(2) {
            paint_segment(
                &mut values,
                to_pixel(pair[0].x),
                to_pixel(pair[0].y),
                to_pixel(pair[1].x),
                to_pixel(pair[1].y),
            );
        }
    }

    values
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::preproc::StrokePoint;

    fn point(x: f64, y: f64) -> StrokePoint {
        StrokePoint {
            x,
            y,
            t: 0.0,
            stroke_index: 0,
        }
    }

    #[test]
    fn rasterize_emits_finite_grid() {
        let strokes = vec![vec![point(0.0, 0.5), point(1.0, 0.5)]];
        let values = rasterize(&strokes);
        assert_eq!(values.len(), 28 * 28);
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().any(|value| *value > 0.0));
    }

    #[test]
    fn rasterize_paints_center_for_crossing_line() {
        let strokes = vec![vec![point(0.0, 0.5), point(1.0, 0.5)]];
        let values = rasterize(&strokes);
        let center = 14 * RASTER_SIZE + 14;
        assert!(values[center] > 0.0);
    }
}
