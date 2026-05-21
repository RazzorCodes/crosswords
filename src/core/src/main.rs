#![allow(dead_code)]

#[cfg(not(target_arch = "wasm32"))]
mod cnn;
#[cfg(not(target_arch = "wasm32"))]
mod data;
#[cfg(not(target_arch = "wasm32"))]
mod logger;
#[cfg(not(target_arch = "wasm32"))]
mod types;

#[cfg(not(target_arch = "wasm32"))]
use std::env;
#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
use burn::backend::NdArray;

#[cfg(not(target_arch = "wasm32"))]
use cnn::model::CnnModelMetadata;
#[cfg(not(target_arch = "wasm32"))]
use cnn::train::{load_safetensors, train_cnn_from_file, TrainCnnOptions};
#[cfg(not(target_arch = "wasm32"))]
use data::dataset::BranchSample;
#[cfg(not(target_arch = "wasm32"))]
use data::pipeline::{preprocess_file, preprocess_strokes_for_cnn};
#[cfg(not(target_arch = "wasm32"))]
use logger::log_progress;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
enum Command {
    Preproc {
        source: PathBuf,
        dest: PathBuf,
    },
    TrainCnn {
        sources: Vec<PathBuf>,
        model_out: PathBuf,
        config_out: PathBuf,
        options: TrainCnnOptions,
    },
    InferCnn {
        model: PathBuf,
        config: PathBuf,
        input: PathBuf,
        top_k: usize,
    },
}

#[cfg(not(target_arch = "wasm32"))]
fn usage() -> &'static str {
    "usage:
  core preproc --source <annotated-json-file> --dest <output-dir>
  core train-cnn --source <annotated-json> [--source <another-json> ...] --model-out <model.safetensors> --config-out <model.json> [--epochs 20] [--batch-size 32] [--lr 0.001] [--seed 42] [--cpu]
  core infer-cnn --model <model.safetensors> --config <model.json> --input <strokes.json> [--top-k 26]"
}

#[cfg(not(target_arch = "wasm32"))]
fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for {flag}\n{}", usage()))
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_usize(value: String, flag: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid {flag} value '{value}': {error}"))
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_f64(value: String, flag: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|error| format!("invalid {flag} value '{value}': {error}"))
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_u64(value: String, flag: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|error| format!("invalid {flag} value '{value}': {error}"))
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_args() -> Result<Command, String> {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        return Err(usage().to_string());
    };

    match command.as_str() {
        "preproc" => {
            let mut source = None;
            let mut dest = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--source" => source = Some(PathBuf::from(next_value(&mut args, "--source")?)),
                    "--dest" => dest = Some(PathBuf::from(next_value(&mut args, "--dest")?)),
                    "--help" | "-h" => return Err(usage().to_string()),
                    other => return Err(format!("unknown argument: {other}\n{}", usage())),
                }
            }
            Ok(Command::Preproc {
                source: source.ok_or_else(|| format!("missing --source\n{}", usage()))?,
                dest: dest.ok_or_else(|| format!("missing --dest\n{}", usage()))?,
            })
        }
        "train-cnn" => {
            let mut sources = Vec::new();
            let mut model_out = None;
            let mut config_out = None;
            let mut options = TrainCnnOptions::default();
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--source" => sources.push(PathBuf::from(next_value(&mut args, "--source")?)),
                    "--model-out" => {
                        model_out = Some(PathBuf::from(next_value(&mut args, "--model-out")?))
                    }
                    "--config-out" => {
                        config_out = Some(PathBuf::from(next_value(&mut args, "--config-out")?))
                    }
                    "--epochs" => {
                        options.epochs =
                            parse_usize(next_value(&mut args, "--epochs")?, "--epochs")?
                    }
                    "--batch-size" => {
                        options.batch_size =
                            parse_usize(next_value(&mut args, "--batch-size")?, "--batch-size")?
                    }
                    "--lr" => {
                        options.learning_rate = parse_f64(next_value(&mut args, "--lr")?, "--lr")?
                    }
                    "--seed" => {
                        options.seed = parse_u64(next_value(&mut args, "--seed")?, "--seed")?
                    }
                    "--cpu" => options.use_cpu = true,
                    "--help" | "-h" => return Err(usage().to_string()),
                    other => return Err(format!("unknown argument: {other}\n{}", usage())),
                }
            }
            if sources.is_empty() {
                return Err(format!("missing --source\n{}", usage()));
            }
            Ok(Command::TrainCnn {
                sources,
                model_out: model_out.ok_or_else(|| format!("missing --model-out\n{}", usage()))?,
                config_out: config_out
                    .ok_or_else(|| format!("missing --config-out\n{}", usage()))?,
                options,
            })
        }
        "infer-cnn" => {
            let mut model = None;
            let mut config = None;
            let mut input = None;
            let mut top_k = 26;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--model" => model = Some(PathBuf::from(next_value(&mut args, "--model")?)),
                    "--config" => config = Some(PathBuf::from(next_value(&mut args, "--config")?)),
                    "--input" => input = Some(PathBuf::from(next_value(&mut args, "--input")?)),
                    "--top-k" => top_k = parse_usize(next_value(&mut args, "--top-k")?, "--top-k")?,
                    "--help" | "-h" => return Err(usage().to_string()),
                    other => return Err(format!("unknown argument: {other}\n{}", usage())),
                }
            }
            Ok(Command::InferCnn {
                model: model.ok_or_else(|| format!("missing --model\n{}", usage()))?,
                config: config.ok_or_else(|| format!("missing --config\n{}", usage()))?,
                input: input.ok_or_else(|| format!("missing --input\n{}", usage()))?,
                top_k,
            })
        }
        "--help" | "-h" => Err(usage().to_string()),
        "--pipeline" => Err(format!(
            "legacy --pipeline syntax was removed; use `core preproc --source ... --dest ...`\n{}",
            usage()
        )),
        other => Err(format!("unknown command: {other}\n{}", usage())),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn write_branch(path: &Path, samples: &[BranchSample]) -> Result<(), String> {
    let text = serde_json::to_string(samples)
        .map_err(|error| format!("failed to serialize {}: {error}", path.display()))?;
    fs::write(path, text).map_err(|error| format!("failed to write {}: {error}", path.display()))
}

#[cfg(not(target_arch = "wasm32"))]
fn run_preproc(source: &Path, dest: &Path) -> Result<(), String> {
    let output = preprocess_file(source)?;
    fs::create_dir_all(dest)
        .map_err(|error| format!("failed to create {}: {error}", dest.display()))?;

    write_branch(&dest.join("1dcnn.json"), &output.one_d_cnn)?;
    write_branch(&dest.join("2dcnn.json"), &output.two_d_cnn)?;
    write_branch(&dest.join("svm.json"), &output.svm)?;
    log_progress("preproc.pipeline", 100, None, "wrote 3 output files");
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn run_infer_cnn(
    model_path: &Path,
    config_path: &Path,
    input_path: &Path,
    top_k: usize,
) -> Result<(), String> {
    type Backend = NdArray<f32>;

    let config_text = fs::read_to_string(config_path)
        .map_err(|error| format!("failed to read {}: {error}", config_path.display()))?;
    let metadata: CnnModelMetadata = serde_json::from_str(&config_text)
        .map_err(|error| format!("failed to parse {}: {error}", config_path.display()))?;
    let input = fs::read_to_string(input_path)
        .map_err(|error| format!("failed to read {}: {error}", input_path.display()))?;
    let sample = preprocess_strokes_for_cnn(&input)?;
    let device = Default::default();
    let model = load_safetensors::<Backend>(&metadata.config(), model_path, &device)?;
    let batch = data::bridge::batcher::batch_dual::<Backend>(&[sample], &device)?;
    let probabilities = model
        .predict_probabilities(batch.one_d, batch.two_d)
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| format!("failed to read probabilities: {error}"))?;

    let mut sorted = probabilities
        .into_iter()
        .enumerate()
        .map(|(index, probability)| {
            let label = metadata
                .labels
                .get(index)
                .cloned()
                .unwrap_or_else(|| ((b'A' + index as u8) as char).to_string());
            (label, probability)
        })
        .collect::<Vec<_>>();
    sorted.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let text = sorted
        .into_iter()
        .take(top_k.min(26))
        .map(|(label, probability)| format!("{label}:{probability:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    println!("{text}");
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn run() -> Result<(), String> {
    match parse_args()? {
        Command::Preproc { source, dest } => run_preproc(&source, &dest),
        Command::TrainCnn {
            sources,
            model_out,
            config_out,
            options,
        } => {
            let metadata = train_cnn_from_file(&sources, &model_out, &config_out, options)?;
            eprintln!(
                "trained {} epochs on {} samples; validation samples: {}",
                metadata.trained_epochs, metadata.train_samples, metadata.validation_samples
            );
            Ok(())
        }
        Command::InferCnn {
            model,
            config,
            input,
            top_k,
        } => run_infer_cnn(&model, &config, &input, top_k),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}
