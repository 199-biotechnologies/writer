//! Punctuation frequency per 1000 words.
use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PunctuationStats {
    pub em_dashes_per_1k: f64,
    pub en_dashes_per_1k: f64,
    pub semicolons_per_1k: f64,
    pub colons_per_1k: f64,
    pub exclamations_per_1k: f64,
    pub questions_per_1k: f64,
    pub parentheses_per_1k: f64,
}

impl PunctuationStats {
    pub fn compute(text: &str) -> Self {
        let word_count = text.unicode_words().count() as f64;
        if word_count == 0.0 {
            return Self::default();
        }

        let per_1k = |count: usize| (count as f64 / word_count) * 1000.0;

        Self {
            em_dashes_per_1k: per_1k(
                text.matches('\u{2014}').count() + text.matches("---").count(),
            ),
            en_dashes_per_1k: per_1k(
                text.matches('\u{2013}').count() + text.matches("--").count()
                    - text.matches("---").count(),
            ), // exclude triple dashes
            semicolons_per_1k: per_1k(text.matches(';').count()),
            colons_per_1k: per_1k(text.matches(':').count()),
            exclamations_per_1k: per_1k(text.matches('!').count()),
            questions_per_1k: per_1k(text.matches('?').count()),
            parentheses_per_1k: per_1k(text.matches('(').count() + text.matches(')').count()),
        }
    }
}
