use chrono::Utc;
use std::path::Path;

pub fn timestamp() -> String {
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

pub fn format_log_line(component: &str, percent: u8, file: Option<&Path>, message: &str) -> String {
    match file {
        Some(file) => format!(
            "[{}] {} [{:03}%] {} {}",
            timestamp(),
            component,
            percent.min(100),
            file.display(),
            message
        ),
        None => format!(
            "[{}] {} [{:03}%] {}",
            timestamp(),
            component,
            percent.min(100),
            message
        ),
    }
}

pub fn log_progress(component: &str, percent: u8, file: Option<&Path>, message: &str) {
    eprintln!("{}", format_log_line(component, percent, file, message));
}
