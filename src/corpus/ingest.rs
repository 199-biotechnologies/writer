//! Top-level ingestion pipeline.
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::corpus::chunker;
use crate::corpus::dedupe;
use crate::corpus::normalize;
use crate::corpus::sample::Sample;
use crate::corpus::sources::{SourceError, SourceRegistry};

#[derive(Debug, Serialize)]
pub struct IngestReport {
    pub samples_added: usize,
    pub samples_skipped_dedupe: usize,
    pub total_words: usize,
    pub contexts: HashMap<String, usize>,
}

#[derive(Debug, thiserror::Error)]
pub enum IngestError {
    #[error("source error: {0}")]
    Source(#[from] SourceError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("no suitable source found for: {0}")]
    NoSource(String),
}

/// Run the full ingestion pipeline: detect → parse → normalize → dedupe → chunk.
pub fn ingest(
    paths: &[PathBuf],
    context: Option<&str>,
    max_seq_len: u32,
    existing_hashes: &HashSet<String>,
    clean: bool,
) -> Result<(Vec<Sample>, IngestReport), IngestError> {
    let registry = SourceRegistry::default_set();
    let max_words = (max_seq_len as usize) / 2; // conservative token estimate

    let mut all_samples: Vec<Sample> = Vec::new();

    for path in paths {
        let samples = parse_path(path, context, &registry)?;
        all_samples.extend(samples);
    }

    // Normalize
    if clean {
        all_samples = all_samples.into_iter().map(normalize::clean).collect();
    }

    // Dedupe
    let dedupe_result = dedupe::dedupe(all_samples, existing_hashes);
    let skipped = dedupe_result.skipped;
    all_samples = dedupe_result.kept;

    // Chunk
    all_samples = chunker::chunk(all_samples, max_words);

    // Build report
    let total_words: usize = all_samples.iter().map(|s| s.word_count()).sum();
    let mut contexts: HashMap<String, usize> = HashMap::new();
    for sample in &all_samples {
        if let Some(tag) = &sample.metadata.context_tag {
            *contexts.entry(tag.clone()).or_insert(0) += 1;
        }
    }

    let report = IngestReport {
        samples_added: all_samples.len(),
        samples_skipped_dedupe: skipped,
        total_words,
        contexts,
    };

    Ok((all_samples, report))
}

fn parse_path(
    path: &Path,
    context: Option<&str>,
    registry: &SourceRegistry,
) -> Result<Vec<Sample>, IngestError> {
    if let Some(source) = registry.detect(path) {
        return Ok(source.parse(path, context)?);
    }

    // If it's a directory, try walking it for markdown files
    if path.is_dir() {
        let mut samples = Vec::new();
        walk_dir_for_files(path, context, registry, &mut samples)?;
        return Ok(samples);
    }

    Err(IngestError::NoSource(path.display().to_string()))
}

fn walk_dir_for_files(
    dir: &Path,
    context: Option<&str>,
    registry: &SourceRegistry,
    samples: &mut Vec<Sample>,
) -> Result<(), IngestError> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if name_str.starts_with('.') {
            continue;
        }

        if path.is_dir() {
            walk_dir_for_files(&path, context, registry, samples)?;
        } else if let Some(source) = registry.detect(&path) {
            match source.parse(&path, context) {
                Ok(parsed) => samples.extend(parsed),
                Err(e) => eprintln!("warning: skipping {}: {e}", path.display()),
            }
        }
    }
    Ok(())
}
