//! Word, sentence, and paragraph length distributions.
use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LengthStats {
    pub mean: f64,
    pub sd: f64,
    pub median: f64,
    pub p95: f64,
}

impl LengthStats {
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let sd = variance.sqrt();

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let p95_idx = ((sorted.len() as f64) * 0.95) as usize;
        let p95 = sorted[p95_idx.min(sorted.len() - 1)];

        Self { mean, sd, median, p95 }
    }
}

pub fn word_lengths(text: &str) -> LengthStats {
    let lengths: Vec<f64> = text
        .unicode_words()
        .map(|w| w.chars().count() as f64)
        .collect();
    LengthStats::from_values(&lengths)
}

pub fn sentence_lengths(text: &str) -> LengthStats {
    let lengths: Vec<f64> = text
        .unicode_sentences()
        .map(|s| s.unicode_words().count() as f64)
        .filter(|&l| l > 0.0)
        .collect();
    LengthStats::from_values(&lengths)
}

pub fn paragraph_lengths(text: &str) -> LengthStats {
    let lengths: Vec<f64> = text
        .split("\n\n")
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .map(|p| p.unicode_words().count() as f64)
        .filter(|&l| l > 0.0)
        .collect();
    LengthStats::from_values(&lengths)
}
