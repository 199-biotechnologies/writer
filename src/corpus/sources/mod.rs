//! Source parsers for different writing formats.
pub mod markdown;
pub mod obsidian;
pub mod plain_text;

use std::path::Path;

use crate::corpus::sample::Sample;

#[derive(Debug, thiserror::Error)]
pub enum SourceError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("unsupported format: {0}")]
    Unsupported(String),
}

pub trait Source: Send + Sync {
    fn name(&self) -> &'static str;
    fn matches(&self, path: &Path) -> bool;
    fn parse(&self, path: &Path, context: Option<&str>) -> Result<Vec<Sample>, SourceError>;
}

pub struct SourceRegistry {
    sources: Vec<Box<dyn Source>>,
}

impl SourceRegistry {
    pub fn default_set() -> Self {
        Self {
            sources: vec![
                Box::new(obsidian::ObsidianSource),
                Box::new(markdown::MarkdownSource),
                Box::new(plain_text::PlainTextSource),
            ],
        }
    }

    pub fn detect(&self, path: &Path) -> Option<&dyn Source> {
        self.sources.iter().find(|s| s.matches(path)).map(|s| s.as_ref())
    }
}
