use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig, GradientsParams, Optimizer};
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn_store::{ModuleSnapshot, SafetensorsStore};
use js_sys::{Float32Array, Uint8Array};
use wasm_bindgen::prelude::*;

use crate::cnn::model::{CnnModelMetadata, PicoDualCnn};
use crate::data::bridge::batcher::{batch_dual, DualSample};
use crate::data::dataset::{RawPoint, RawSample};
use crate::data::pipeline::{preprocess_sample_for_cnn, preprocess_strokes_for_cnn};
use crate::types::ALPHABET_LEN;

type WasmBackend = Autodiff<NdArray<f32>>;
type WasmOptimizer = OptimizerAdaptor<AdamW, PicoDualCnn<WasmBackend>, WasmBackend>;

const BATCH_MAGIC: &[u8; 4] = b"BCNN";
const BATCH_VERSION: u32 = 1;
const DEFAULT_EPOCHS: usize = 4;
const DEFAULT_BATCH_SIZE: usize = 8;
const DEFAULT_LEARNING_RATE: f64 = 0.001;

#[wasm_bindgen]
pub struct FineTuneResult {
    model_bytes: Vec<u8>,
    optimizer_bytes: Vec<u8>,
    metadata_json: String,
    average_loss: f64,
    trained_samples: usize,
}

#[wasm_bindgen]
impl FineTuneResult {
    #[wasm_bindgen(getter)]
    pub fn model_bytes(&self) -> Uint8Array {
        Uint8Array::from(self.model_bytes.as_slice())
    }

    #[wasm_bindgen(getter)]
    pub fn optimizer_bytes(&self) -> Uint8Array {
        Uint8Array::from(self.optimizer_bytes.as_slice())
    }

    #[wasm_bindgen(getter)]
    pub fn metadata_json(&self) -> String {
        self.metadata_json.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn average_loss(&self) -> f64 {
        self.average_loss
    }

    #[wasm_bindgen(getter)]
    pub fn trained_samples(&self) -> usize {
        self.trained_samples
    }
}

#[wasm_bindgen]
pub struct WasmCnn {
    model: PicoDualCnn<WasmBackend>,
    optimizer: WasmOptimizer,
    metadata: CnnModelMetadata,
    device: NdArrayDevice,
}

#[wasm_bindgen]
impl WasmCnn {
    pub async fn from_safetensors(
        model_bytes: Uint8Array,
        config_json: String,
        optimizer_bytes: Option<Uint8Array>,
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
        model = model.freeze_backbone();

        let mut optimizer = AdamWConfig::new().init();
        if let Some(bytes) = optimizer_bytes {
            if bytes.length() > 0 {
                let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
                let record = recorder
                    .load(bytes.to_vec(), &device)
                    .map_err(|error| {
                        JsValue::from_str(&format!("failed to load optimizer state: {error}"))
                    })?;
                optimizer = optimizer.load_record(record);
            }
        }

        Ok(WasmCnn {
            model,
            optimizer,
            metadata,
            device,
        })
    }

    pub async fn predict_strokes(&self, strokes_json: String) -> Result<Float32Array, JsValue> {
        let sample =
            preprocess_strokes_for_cnn(&strokes_json).map_err(|error| JsValue::from_str(&error))?;
        let batch = batch_dual::<WasmBackend>(&[sample], &self.device)
            .map_err(|error| JsValue::from_str(&error))?;
        let values = self
            .model
            .valid()
            .predict_probabilities(batch.one_d.inner(), batch.two_d.inner())
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| {
                JsValue::from_str(&format!("failed to read probabilities: {error}"))
            })?;
        Ok(Float32Array::from(values.as_slice()))
    }

    pub async fn fine_tune_head(
        &mut self,
        batch_bytes: Uint8Array,
        epochs: Option<usize>,
        lr: Option<f64>,
        batch_size: Option<usize>,
    ) -> Result<FineTuneResult, JsValue> {
        let samples = decode_training_batch(&batch_bytes.to_vec())
            .map_err(|error| JsValue::from_str(&error))?;
        if samples.is_empty() {
            return Err(JsValue::from_str("training batch has no usable samples"));
        }

        let epochs = epochs.unwrap_or(DEFAULT_EPOCHS).max(1);
        let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE).max(1);
        let learning_rate = lr.unwrap_or(DEFAULT_LEARNING_RATE);
        let mut loss_sum = 0.0;
        let mut loss_count = 0_usize;

        for _ in 0..epochs {
            for chunk in samples.chunks(batch_size) {
                let batch = batch_dual::<WasmBackend>(chunk, &self.device)
                    .map_err(|error| JsValue::from_str(&error))?;
                let model = self.model.clone().freeze_backbone();
                let logits = model.forward(batch.one_d, batch.two_d);
                let loss = CrossEntropyLossConfig::new()
                    .with_logits(true)
                    .init(&logits.device())
                    .forward(logits, batch.labels);
                if let Ok(values) = loss.clone().into_data().to_vec::<f32>() {
                    if let Some(value) = values.first() {
                        loss_sum += *value as f64;
                        loss_count += 1;
                    }
                }
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                self.model = self
                    .optimizer
                    .step(learning_rate, model, grads)
                    .freeze_backbone();
            }
        }

        self.metadata.trained_epochs += epochs;
        self.metadata.batch_size = batch_size;
        self.metadata.learning_rate = learning_rate;
        self.metadata.train_samples = samples.len();

        let model_bytes = self.export_model_bytes()?;
        let optimizer_bytes = self.export_optimizer_bytes()?;
        let metadata_json = serde_json::to_string(&self.metadata).map_err(|error| {
            JsValue::from_str(&format!("failed to serialize model metadata: {error}"))
        })?;

        Ok(FineTuneResult {
            model_bytes,
            optimizer_bytes,
            metadata_json,
            average_loss: if loss_count > 0 {
                loss_sum / loss_count as f64
            } else {
                0.0
            },
            trained_samples: samples.len(),
        })
    }
}

impl WasmCnn {
    fn export_model_bytes(&self) -> Result<Vec<u8>, JsValue> {
        let model = self.model.valid();
        let mut store = SafetensorsStore::from_bytes(None).overwrite(true);
        model
            .save_into(&mut store)
            .map_err(|error| JsValue::from_str(&format!("failed to export model: {error}")))?;
        store
            .get_bytes()
            .map_err(|error| JsValue::from_str(&format!("failed to read model bytes: {error}")))
    }

    fn export_optimizer_bytes(&self) -> Result<Vec<u8>, JsValue> {
        BinBytesRecorder::<FullPrecisionSettings>::default()
            .record(self.optimizer.to_record(), ())
            .map_err(|error| JsValue::from_str(&format!("failed to export optimizer: {error}")))
    }
}

fn decode_training_batch(bytes: &[u8]) -> Result<Vec<DualSample>, String> {
    let mut reader = BatchReader { bytes, offset: 0 };
    if reader.read_bytes(4)? != BATCH_MAGIC {
        return Err("invalid CNN batch magic".to_string());
    }
    let version = reader.read_u32()?;
    if version != BATCH_VERSION {
        return Err(format!("unsupported CNN batch version {version}"));
    }
    let sample_count = reader.read_u32()? as usize;
    let mut samples = Vec::with_capacity(sample_count);

    for sample_index in 0..sample_count {
        let label_index = reader.read_u8()? as usize;
        if label_index >= ALPHABET_LEN {
            return Err(format!("sample {sample_index} has invalid label {label_index}"));
        }
        let stroke_count = reader.read_u32()? as usize;
        let mut strokes = Vec::with_capacity(stroke_count);
        for _ in 0..stroke_count {
            let point_count = reader.read_u32()? as usize;
            let mut stroke = Vec::with_capacity(point_count);
            for _ in 0..point_count {
                stroke.push(RawPoint {
                    x: reader.read_f32()? as f64,
                    y: reader.read_f32()? as f64,
                    t: reader.read_f32()? as f64,
                });
            }
            strokes.push(stroke);
        }
        let raw = RawSample {
            sample_id: format!("wasm-{sample_index}"),
            sample_label: ((b'A' + label_index as u8) as char).to_string(),
            extra_labels: vec![],
            strokes,
        };
        samples.push(preprocess_sample_for_cnn(&raw)?);
    }

    if reader.offset != bytes.len() {
        return Err("CNN batch has trailing bytes".to_string());
    }

    Ok(samples)
}

struct BatchReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> BatchReader<'a> {
    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], String> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| "CNN batch offset overflow".to_string())?;
        if end > self.bytes.len() {
            return Err("CNN batch ended unexpectedly".to_string());
        }
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        let bytes: [u8; 4] = self.read_bytes(4)?.try_into().unwrap();
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_f32(&mut self) -> Result<f32, String> {
        let bytes: [u8; 4] = self.read_bytes(4)?.try_into().unwrap();
        let value = f32::from_le_bytes(bytes);
        if value.is_finite() {
            Ok(value)
        } else {
            Err("CNN batch contains non-finite point value".to_string())
        }
    }
}
