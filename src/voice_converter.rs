//! Voice cache conversion from PyTorch (.pt) to Safetensors format.
//!
//! This module provides functionality to convert voice cache files used by
//! VibeVoice Realtime (0.5B) from PyTorch checkpoint format to Safetensors
//! format that can be loaded efficiently by Rust.
//!
//! # Usage
//!
//! ```bash
//! # Convert all .pt files in voices/streaming_model (default)
//! vibevoice-rs convert-voice
//!
//! # Convert single file
//! vibevoice-rs convert-voice voice.pt voice.safetensors
//! ```

use anyhow::{Result, anyhow};
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::info;

/// Default directory containing voice cache files.
pub const DEFAULT_VOICE_DIR: &str = "voices/streaming_model";

/// Subdirectory for archiving original .pt files after conversion.
pub const ARCHIVE_SUBDIR: &str = "pt_archive";

/// Result of a batch conversion operation.
#[derive(Debug)]
pub struct ConversionReport {
    /// Number of files successfully converted.
    pub converted: usize,
    /// Number of files that failed to convert.
    pub failed: usize,
    /// Paths of successfully converted files.
    pub success_paths: Vec<PathBuf>,
    /// Paths and errors of failed conversions.
    pub failures: Vec<(PathBuf, String)>,
}

impl ConversionReport {
    fn new() -> Self {
        Self {
            converted: 0,
            failed: 0,
            success_paths: Vec::new(),
            failures: Vec::new(),
        }
    }

    fn add_success(&mut self, path: PathBuf) {
        self.converted += 1;
        self.success_paths.push(path);
    }

    fn add_failure(&mut self, path: PathBuf, error: String) {
        self.failed += 1;
        self.failures.push((path, error));
    }
}

/// Convert voice cache file(s) from .pt to .safetensors format.
///
/// # Arguments
///
/// * `input` - Optional input .pt file path. If None, converts all files in default directory.
/// * `output` - Optional output .safetensors file path. If None, auto-generates from input name.
///
/// # Returns
///
/// A `ConversionReport` containing the results of the conversion operation.
pub fn convert_voice_caches(
    input: Option<&Path>,
    output: Option<&Path>,
) -> Result<ConversionReport> {
    match (input, output) {
        (Some(inp), Some(out)) => {
            let mut report = ConversionReport::new();
            match convert_single(inp, out) {
                Ok(()) => report.add_success(out.to_path_buf()),
                Err(e) => report.add_failure(inp.to_path_buf(), e.to_string()),
            }
            Ok(report)
        }
        (Some(inp), None) => {
            let out = inp.with_extension("safetensors");
            let mut report = ConversionReport::new();
            match convert_single(inp, &out) {
                Ok(()) => report.add_success(out),
                Err(e) => report.add_failure(inp.to_path_buf(), e.to_string()),
            }
            Ok(report)
        }
        (None, _) => convert_all_and_archive(),
    }
}

/// Convert all .pt files in the default voice directory and archive originals.
///
/// This function:
/// 1. Scans `voices/streaming_model` for `.pt` files
/// 2. Converts each to `.safetensors` format in the same directory
/// 3. Moves the original `.pt` files to `voices/streaming_model/pt_archive/`
fn convert_all_and_archive() -> Result<ConversionReport> {
    let voice_dir = PathBuf::from(DEFAULT_VOICE_DIR);
    let archive_dir = voice_dir.join(ARCHIVE_SUBDIR);

    // Verify voice directory exists
    if !voice_dir.exists() {
        return Err(anyhow!(
            "Voice directory '{}' does not exist",
            voice_dir.display()
        ));
    }

    // Create archive directory
    std::fs::create_dir_all(&archive_dir)?;

    // Find all .pt files (exclude already archived files)
    let pt_files: Vec<_> = std::fs::read_dir(&voice_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let path = e.path();
            path.is_file() && path.extension().map(|x| x == "pt").unwrap_or(false)
        })
        .collect();

    if pt_files.is_empty() {
        info!("No .pt files found in {}", voice_dir.display());
        return Ok(ConversionReport::new());
    }

    info!("Found {} .pt files to convert", pt_files.len());

    let mut report = ConversionReport::new();

    for entry in pt_files {
        let input = entry.path();
        let output = input.with_extension("safetensors");
        let filename = input.file_name().unwrap();

        info!("Converting {:?}...", filename);

        match convert_single(&input, &output) {
            Ok(()) => {
                // Move original to archive
                let archive_path = archive_dir.join(filename);
                if let Err(e) = std::fs::rename(&input, &archive_path) {
                    report.add_failure(
                        input.clone(),
                        format!("Converted but failed to archive: {}", e),
                    );
                } else {
                    info!("  Archived to {:?}", archive_path.file_name().unwrap());
                    report.add_success(output);
                }
            }
            Err(e) => {
                report.add_failure(input, e.to_string());
            }
        }
    }

    Ok(report)
}

/// Convert a single .pt file to .safetensors format using the Python script.
///
/// Attempts to use `uv run` first, falling back to `python3` if unavailable.
fn convert_single(input: &Path, output: &Path) -> Result<()> {
    // Verify input exists
    if !input.exists() {
        return Err(anyhow!("Input file does not exist: {}", input.display()));
    }

    // Try uv first
    let uv_result = Command::new("uv")
        .args(["run", "scripts/convert_voice_cache.py"])
        .arg(input)
        .arg(output)
        .output();

    match uv_result {
        Ok(out) if out.status.success() => return Ok(()),
        Ok(out) => {
            // uv ran but script failed - check if it's a "uv not found" vs script error
            let stderr = String::from_utf8_lossy(&out.stderr);
            // If uv ran the script but it failed, report the error
            if !stderr.trim().is_empty() && !stderr.contains("No such file") {
                return Err(anyhow!("Conversion failed: {}", stderr.trim()));
            }
            // Otherwise fall through to python3 fallback
        }
        Err(_) => {
            // uv not available, fall through to python3
        }
    }

    // Fall back to python3
    let py_result = Command::new("python3")
        .arg("scripts/convert_voice_cache.py")
        .arg(input)
        .arg(output)
        .output()?;

    if !py_result.status.success() {
        let stderr = String::from_utf8_lossy(&py_result.stderr);
        return Err(anyhow!("Conversion failed: {}", stderr.trim()));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversion_report() {
        let mut report = ConversionReport::new();
        assert_eq!(report.converted, 0);
        assert_eq!(report.failed, 0);

        report.add_success(PathBuf::from("test.safetensors"));
        assert_eq!(report.converted, 1);
        assert_eq!(report.success_paths.len(), 1);

        report.add_failure(PathBuf::from("bad.pt"), "error".to_string());
        assert_eq!(report.failed, 1);
        assert_eq!(report.failures.len(), 1);
    }

    #[test]
    fn test_default_paths() {
        assert_eq!(DEFAULT_VOICE_DIR, "voices/streaming_model");
        assert_eq!(ARCHIVE_SUBDIR, "pt_archive");
    }
}
