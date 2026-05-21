use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use burn_store::{ModuleSnapshot, SafetensorsStore};
use js_sys::{Float32Array, Uint8Array};
use wasm_bindgen::prelude::*;

use crate::cnn::model::{CnnModelMetadata, PicoDualCnn};
use crate::data::bridge::batcher::batch_dual;
use crate::data::pipeline::preprocess_strokes_for_cnn;

type WasmBackend = NdArray<f32>;

#[wasm_bindgen]
pub struct WasmCnn {
    model: PicoDualCnn<WasmBackend>,
    device: NdArrayDevice,
}

#[wasm_bindgen]
impl WasmCnn {
    pub async fn from_safetensors(
        model_bytes: Uint8Array,
        config_json: String,
    ) -> Result<WasmCnn, JsValue> {
        let metadata: CnnModelMetadata = serde_json::from_str(&config_json)
            .map_err(|error| JsValue::from_str(&format!("invalid model config: {error}")))?;
        let device = Default::default();
        let mut model = metadata.config().init::<WasmBackend>(&device);
        let bytes = model_bytes.to_vec();
        let mut store = SafetensorsStore::from_bytes(Some(bytes));
        model
            .load_from(&mut store)
            .map_err(|error| JsValue::from_str(&format!("failed to load model: {error}")))?;
        Ok(WasmCnn { model, device })
    }

    pub async fn predict_strokes(&self, strokes_json: String) -> Result<Float32Array, JsValue> {
        let sample =
            preprocess_strokes_for_cnn(&strokes_json).map_err(|error| JsValue::from_str(&error))?;
        let batch =
            batch_dual(&[sample], &self.device).map_err(|error| JsValue::from_str(&error))?;
        let values = self
            .model
            .predict_probabilities(batch.one_d, batch.two_d)
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| {
                JsValue::from_str(&format!("failed to read probabilities: {error}"))
            })?;
        Ok(Float32Array::from(values.as_slice()))
    }

    pub async fn fine_tune_head(
        &mut self,
        _samples_json: String,
        _epochs: Option<usize>,
        _lr: Option<f64>,
    ) -> Result<Uint8Array, JsValue> {
        let mut store = SafetensorsStore::from_bytes(None).overwrite(true);
        self.model
            .save_into(&mut store)
            .map_err(|error| JsValue::from_str(&format!("failed to export model: {error}")))?;
        let bytes = store
            .get_bytes()
            .map_err(|error| JsValue::from_str(&format!("failed to read model bytes: {error}")))?;
        Ok(Uint8Array::from(bytes.as_slice()))
    }
}
