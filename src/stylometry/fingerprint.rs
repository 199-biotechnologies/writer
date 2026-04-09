//! Stylometric fingerprint computation.
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::corpus::sample::Sample;
use crate::stylometry::features::function_words;
use crate::stylometry::features::lengths::{self, LengthStats};
use crate::stylometry::features::ngrams;
use crate::stylometry::features::punctuation::PunctuationStats;
use crate::stylometry::features::vocabulary;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StylometricFingerprint {
    pub word_count: u64,
    pub char_count: u64,
    pub word_length: LengthStats,
    pub sentence_length: LengthStats,
    pub paragraph_length: LengthStats,
    pub function_words: HashMap<String, f64>,
    pub ngram_profile: Vec<(String, u64)>,
    pub punctuation: PunctuationStats,
    pub vocabulary_size: u64,
    pub banned_words: Vec<String>,
    pub preferred_words: Vec<(String, f64)>,
}

impl StylometricFingerprint {
    /// Compute a fingerprint from a collection of writing samples.
    pub fn compute(samples: &[Sample]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }

        // Concatenate all sample content
        let full_text: String = samples
            .iter()
            .map(|s| s.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");

        let word_count = samples.iter().map(|s| s.word_count() as u64).sum();
        let char_count = samples.iter().map(|s| s.char_count() as u64).sum();

        let word_length = lengths::word_lengths(&full_text);
        let sentence_length = lengths::sentence_lengths(&full_text);
        let paragraph_length = lengths::paragraph_lengths(&full_text);
        let function_words_freq = function_words::compute(&full_text);
        let trigram_profile = ngrams::trigrams(&full_text);
        let punctuation = PunctuationStats::compute(&full_text);
        let vocab_analysis = vocabulary::analyze(&full_text);

        Self {
            word_count,
            char_count,
            word_length,
            sentence_length,
            paragraph_length,
            function_words: function_words_freq,
            ngram_profile: trigram_profile,
            punctuation,
            vocabulary_size: vocab_analysis.vocabulary_size,
            banned_words: vocab_analysis.banned_words,
            preferred_words: vocab_analysis.preferred_words,
        }
    }
}
