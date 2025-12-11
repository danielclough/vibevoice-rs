use anyhow::{Result, anyhow};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Voice mapper for mapping speaker names to voice file paths
/// Matches Python: VoiceMapper class
pub struct VoiceMapper {
    voice_presets: HashMap<String, PathBuf>,
}

impl VoiceMapper {
    /// Create a new VoiceMapper by scanning a voices directory
    /// Matches Python: VoiceMapper.__init__()
    pub fn new(voices_dir: &Path) -> Result<Self> {
        info!("🎤 Initializing VoiceMapper...");

        let mut voice_presets = HashMap::new();

        // Scan directory for .wav files
        if voices_dir.exists() && voices_dir.is_dir() {
            for entry in fs::read_dir(voices_dir)? {
                let entry = entry?;
                let path = entry.path();

                if path.extension().and_then(|s| s.to_str()) == Some("wav")
                    && let Some(file_stem) = path.file_stem().and_then(|s| s.to_str())
                {
                    // Extract voice name (handle formats like "en-Frank_man.wav")
                    let name = Self::extract_voice_name(file_stem);
                    voice_presets.insert(name.clone(), path.clone());
                    debug!(
                        "   Found voice: {} -> {:?}",
                        name,
                        path.file_name().unwrap()
                    );
                }
            }
        }

        info!("✓ Loaded {} voice presets", voice_presets.len());

        Ok(Self { voice_presets })
    }

    /// Extract voice name from filename
    /// Handles formats like "en-Frank_man" -> "Frank"
    fn extract_voice_name(file_stem: &str) -> String {
        // Split on common separators
        let parts: Vec<&str> = file_stem.split(['-', '_']).collect();

        // Look for the name part (usually second element after language code)
        if parts.len() >= 2 {
            parts[1].to_string()
        } else {
            file_stem.to_string()
        }
    }

    /// Get voice path for a speaker name
    /// Matches Python: VoiceMapper.get_voice_path()
    pub fn get_voice_path(&self, speaker_name: &str) -> Result<PathBuf> {
        // Try exact match first
        if let Some(path) = self.voice_presets.get(speaker_name) {
            return Ok(path.clone());
        }

        // Try case-insensitive match
        let speaker_lower = speaker_name.to_lowercase();
        for (name, path) in &self.voice_presets {
            if name.to_lowercase() == speaker_lower {
                return Ok(path.clone());
            }
        }

        // Try partial match
        for (name, path) in &self.voice_presets {
            if name.to_lowercase().contains(&speaker_lower)
                || speaker_lower.contains(&name.to_lowercase())
            {
                return Ok(path.clone());
            }
        }

        // Fallback to first voice (alphabetically sorted) if available
        // Match Python: uses first voice from sorted dict
        let mut sorted_voices: Vec<_> = self.voice_presets.iter().collect();
        sorted_voices.sort_by_key(|(name, _)| *name);

        if let Some((_name, path)) = sorted_voices.first() {
            warn!(
                "No match for '{}', using fallback voice: {:?}",
                speaker_name,
                path.file_name().unwrap()
            );
            return Ok((*path).clone());
        }

        // No voices available at all
        Err(anyhow!("No voice files found in voices directory"))
    }

    /// Map speaker numbers to voice sample paths
    /// Returns a list of voice paths for each UNIQUE speaker (deduplicates like Python)
    pub fn map_speakers_to_voices(&self, speaker_numbers: &[String]) -> Result<Vec<Vec<PathBuf>>> {
        info!("🗺️  Mapping speakers to voices...");

        // Deduplicate while preserving order (like Python's set logic)
        let mut seen = HashSet::new();
        let unique_speakers: Vec<_> = speaker_numbers
            .iter()
            .filter(|s| seen.insert(s.to_string()))
            .collect();

        info!(
            "   Found {} unique speakers from {} total segments",
            unique_speakers.len(),
            speaker_numbers.len()
        );

        let mut voice_samples = Vec::new();
        for speaker_num in unique_speakers {
            let voice_path = self.get_voice_path(speaker_num)?;
            debug!(
                "   Speaker {} -> {:?}",
                speaker_num,
                voice_path.file_name().unwrap()
            );
            voice_samples.push(vec![voice_path]);
        }

        Ok(voice_samples)
    }
}

/// Parse a script file in "Speaker X: text" format
/// Matches Python: parse_txt_script(txt_content)
/// Returns (scripts, speaker_numbers) where scripts KEEP the "Speaker X:" prefix
pub fn parse_txt_script(content: &str) -> Result<(Vec<String>, Vec<String>)> {
    info!("📝 Parsing script...");

    // Match Python regex: r'^Speaker\s+(\d+):\s*(.*)$'
    let re = Regex::new(r"^Speaker\s+(\d+):\s*(.*)$")
        .map_err(|e| anyhow!("Failed to compile regex: {}", e))?;

    let mut scripts = Vec::new();
    let mut speaker_numbers = Vec::new();

    let mut current_speaker: Option<String> = None;
    let mut current_text = String::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(caps) = re.captures(line) {
            // Save previous speaker's accumulated text
            if let Some(prev_speaker) = current_speaker {
                let full_line = format!("Speaker {}: {}", prev_speaker, current_text.trim());
                scripts.push(full_line);
                speaker_numbers.push(prev_speaker);
            }

            // Start new speaker
            current_speaker = Some(caps.get(1).unwrap().as_str().to_string());
            current_text = caps.get(2).unwrap().as_str().to_string();
        } else if current_speaker.is_some() {
            // Continue text for current speaker (multi-line support)
            if !current_text.is_empty() {
                current_text.push(' ');
            }
            current_text.push_str(line);
        }
    }

    // Don't forget the last speaker
    if let Some(speaker) = current_speaker {
        let full_line = format!("Speaker {}: {}", speaker, current_text.trim());
        scripts.push(full_line);
        speaker_numbers.push(speaker);
    }

    info!("✓ Parsed {} speaker lines", scripts.len());

    Ok((scripts, speaker_numbers))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_script() {
        let content = "Speaker 1: Hello world\nSpeaker 2: How are you?";
        let (scripts, speaker_numbers) = parse_txt_script(content).unwrap();

        assert_eq!(scripts.len(), 2);
        assert_eq!(speaker_numbers.len(), 2);
        assert_eq!(scripts[0], "Speaker 1: Hello world");
        assert_eq!(scripts[1], "Speaker 2: How are you?");
        assert_eq!(speaker_numbers[0], "1");
        assert_eq!(speaker_numbers[1], "2");
    }

    #[test]
    fn test_extract_voice_name() {
        assert_eq!(VoiceMapper::extract_voice_name("en-Frank_man"), "Frank");
        assert_eq!(VoiceMapper::extract_voice_name("en-Alice_woman"), "Alice");
        assert_eq!(VoiceMapper::extract_voice_name("simple"), "simple");
    }
}
