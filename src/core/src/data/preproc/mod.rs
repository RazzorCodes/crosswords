pub mod raster;
pub mod sequence;
pub mod spatial;
pub mod svm;
pub mod temporal;

pub const RESAMPLED_POINTS: usize = 64;
pub const CNN_1D_CHANNELS: usize = 3;
pub const RASTER_SIZE: usize = 28;
pub const SVM_FEATURES: usize = RESAMPLED_POINTS * 2 + 3;

#[derive(Clone, Debug, PartialEq)]
pub struct StrokePoint {
    pub x: f64,
    pub y: f64,
    pub t: f64,
    pub stroke_index: usize,
}

pub type StrokePath = Vec<Vec<StrokePoint>>;
