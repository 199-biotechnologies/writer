//! Content-hash deduplication.
use std::collections::HashSet;

use crate::corpus::sample::Sample;

pub struct DedupeResult {
    pub kept: Vec<Sample>,
    pub skipped: usize,
}

/// Deduplicate samples by content hash against existing hashes.
pub fn dedupe(samples: Vec<Sample>, existing_hashes: &HashSet<String>) -> DedupeResult {
    let mut seen = existing_hashes.clone();
    let mut kept = Vec::new();
    let mut skipped = 0;

    for sample in samples {
        if seen.contains(&sample.content_hash) {
            skipped += 1;
        } else {
            seen.insert(sample.content_hash.clone());
            kept.push(sample);
        }
    }

    DedupeResult { kept, skipped }
}
