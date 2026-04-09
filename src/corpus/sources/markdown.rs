//! Markdown source parser.
use std::path::Path;

use crate::corpus::sample::{Sample, SampleMetadata, SampleSource};

use super::{Source, SourceError};

pub struct MarkdownSource;

impl Source for MarkdownSource {
    fn name(&self) -> &'static str {
        "markdown"
    }

    fn matches(&self, path: &Path) -> bool {
        path.extension().is_some_and(|e| e == "md" || e == "markdown")
    }

    fn parse(&self, path: &Path, context: Option<&str>) -> Result<Vec<Sample>, SourceError> {
        let content = std::fs::read_to_string(path)?;
        Ok(parse_markdown(&content, path, context))
    }
}

pub fn parse_markdown(content: &str, origin: &Path, context: Option<&str>) -> Vec<Sample> {
    let body = strip_front_matter(content);
    let body = strip_html_comments(&body);

    let sections = split_by_headers(&body);

    sections
        .into_iter()
        .filter(|(_, text)| {
            let word_count = text.split_whitespace().count();
            word_count >= 5
        })
        .map(|(header, text)| {
            let tag = context
                .map(|c| c.to_string())
                .or_else(|| header.map(|h| h.to_string()))
                .unwrap_or_else(|| "longform".to_string());

            Sample::new(
                text,
                SampleMetadata {
                    source: SampleSource::Markdown,
                    origin_path: Some(origin.to_path_buf()),
                    context_tag: Some(tag),
                    captured_at: None,
                },
            )
        })
        .collect()
}

fn strip_front_matter(content: &str) -> String {
    if content.starts_with("---\n") || content.starts_with("---\r\n") {
        if let Some(end) = content[3..].find("\n---") {
            let after = end + 3 + 4; // skip past closing ---\n
            return content[after..].trim_start_matches('\n').to_string();
        }
    }
    content.to_string()
}

fn strip_html_comments(content: &str) -> String {
    let mut result = String::with_capacity(content.len());
    let mut i = 0;
    let bytes = content.as_bytes();
    while i < bytes.len() {
        if i + 3 < bytes.len() && &content[i..i + 4] == "<!--" {
            if let Some(end) = content[i + 4..].find("-->") {
                i = i + 4 + end + 3;
                continue;
            }
        }
        result.push(bytes[i] as char);
        i += 1;
    }
    result
}

fn split_by_headers(content: &str) -> Vec<(Option<&str>, String)> {
    let mut sections: Vec<(Option<&str>, String)> = Vec::new();
    let mut current_header: Option<&str> = None;
    let mut current_body = String::new();

    for line in content.lines() {
        if line.starts_with("# ") && !line.starts_with("##") {
            if !current_body.trim().is_empty() {
                sections.push((current_header, current_body.trim().to_string()));
            }
            current_header = Some(line[2..].trim());
            current_body = String::new();
        } else {
            current_body.push_str(line);
            current_body.push('\n');
        }
    }

    if !current_body.trim().is_empty() {
        sections.push((current_header, current_body.trim().to_string()));
    }

    sections
}
