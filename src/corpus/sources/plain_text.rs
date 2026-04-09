//! Plain text source parser.
use std::path::Path;

use crate::corpus::sample::{Sample, SampleMetadata, SampleSource};

use super::{Source, SourceError};

pub struct PlainTextSource;

impl Source for PlainTextSource {
    fn name(&self) -> &'static str {
        "plain_text"
    }

    fn matches(&self, path: &Path) -> bool {
        path.extension().is_some_and(|e| e == "txt" || e == "text")
    }

    fn parse(&self, path: &Path, context: Option<&str>) -> Result<Vec<Sample>, SourceError> {
        let content = std::fs::read_to_string(path)?;
        Ok(parse_plain_text(&content, path, context))
    }
}

pub fn parse_plain_text(content: &str, origin: &Path, context: Option<&str>) -> Vec<Sample> {
    let content = strip_bom(content);
    let content = normalize_line_endings(&content);

    let paragraphs: Vec<&str> = content
        .split("\n\n")
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect();

    let mut samples = Vec::new();
    let mut current_chunk = String::new();
    let max_words = 1024;

    for para in paragraphs {
        let chunk_words = current_chunk.split_whitespace().count();
        let para_words = para.split_whitespace().count();

        if chunk_words + para_words > max_words && !current_chunk.is_empty() {
            samples.push(make_sample(
                current_chunk.trim().to_string(),
                origin,
                context,
            ));
            current_chunk = String::new();
        }
        if !current_chunk.is_empty() {
            current_chunk.push_str("\n\n");
        }
        current_chunk.push_str(para);
    }

    if !current_chunk.trim().is_empty() && current_chunk.split_whitespace().count() >= 5 {
        samples.push(make_sample(
            current_chunk.trim().to_string(),
            origin,
            context,
        ));
    }

    samples
}

fn make_sample(content: String, origin: &Path, context: Option<&str>) -> Sample {
    Sample::new(
        content,
        SampleMetadata {
            source: SampleSource::PlainText,
            origin_path: Some(origin.to_path_buf()),
            context_tag: context.map(|c| c.to_string()),
            captured_at: None,
        },
    )
}

fn strip_bom(content: &str) -> String {
    content.strip_prefix('\u{FEFF}').unwrap_or(content).to_string()
}

fn normalize_line_endings(content: &str) -> String {
    content.replace("\r\n", "\n").replace('\r', "\n")
}
