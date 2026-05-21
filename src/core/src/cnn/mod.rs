pub mod model;
#[cfg(not(target_arch = "wasm32"))]
pub mod train;

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
pub mod wasm;
