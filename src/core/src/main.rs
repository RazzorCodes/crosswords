#![allow(dead_code)]

mod data;
mod types;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use data::dataset::BranchSample;
use data::pipeline::{log_progress, preprocess_file};

#[derive(Debug)]
struct Cli {
    pipeline: String,
    source: PathBuf,
    dest: PathBuf,
}

fn usage() -> &'static str {
    "usage: core --pipeline preproc --source <annotated-json-file> --dest <output-dir>"
}

fn parse_args() -> Result<Cli, String> {
    let mut args = env::args().skip(1);
    let mut pipeline = None;
    let mut source = None;
    let mut dest = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--pipeline" => pipeline = args.next(),
            "--source" => source = args.next().map(PathBuf::from),
            "--dest" => dest = args.next().map(PathBuf::from),
            "--help" | "-h" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument: {other}\n{}", usage())),
        }
    }

    Ok(Cli {
        pipeline: pipeline.ok_or_else(|| format!("missing --pipeline\n{}", usage()))?,
        source: source.ok_or_else(|| format!("missing --source\n{}", usage()))?,
        dest: dest.ok_or_else(|| format!("missing --dest\n{}", usage()))?,
    })
}

fn write_branch(path: &Path, samples: &[BranchSample]) -> Result<(), String> {
    let text = serde_json::to_string(samples)
        .map_err(|error| format!("failed to serialize {}: {error}", path.display()))?;
    fs::write(path, text).map_err(|error| format!("failed to write {}: {error}", path.display()))
}

fn run() -> Result<(), String> {
    let cli = parse_args()?;
    if cli.pipeline != "preproc" {
        return Err(format!(
            "unsupported pipeline '{}'; expected 'preproc'",
            cli.pipeline
        ));
    }

    let output = preprocess_file(&cli.source)?;
    fs::create_dir_all(&cli.dest)
        .map_err(|error| format!("failed to create {}: {error}", cli.dest.display()))?;

    write_branch(&cli.dest.join("1dcnn.json"), &output.one_d_cnn)?;
    write_branch(&cli.dest.join("2dcnn.json"), &output.two_d_cnn)?;
    write_branch(&cli.dest.join("svm.json"), &output.svm)?;
    log_progress("preproc.pipeline", 100, None, "wrote 3 output files");

    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
