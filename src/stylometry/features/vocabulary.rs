//! User vocabulary analysis — banned and preferred word lists.
use std::collections::{HashMap, HashSet};

use unicode_segmentation::UnicodeSegmentation;

use crate::stylometry::ai_slop;

/// Compute the user's vocabulary set and derive banned/preferred lists.
pub fn analyze(text: &str) -> VocabularyAnalysis {
    let words: Vec<String> = text
        .unicode_words()
        .map(|w| w.to_lowercase())
        .collect();

    let total = words.len();
    let user_vocab: HashSet<String> = words.iter().cloned().collect();

    // Count word frequencies
    let mut freq: HashMap<String, usize> = HashMap::new();
    for word in &words {
        *freq.entry(word.clone()).or_insert(0) += 1;
    }

    // Banned: AI-slop words NOT in user vocabulary
    let banned: Vec<String> = ai_slop::BANNED_WORDS
        .iter()
        .filter(|w| !user_vocab.contains(**w))
        .map(|w| w.to_string())
        .collect();

    // Preferred: words the user uses unusually often (> 0.1% of corpus, > 3 occurrences)
    let preferred: Vec<(String, f64)> = freq
        .iter()
        .filter(|(_, count)| **count > 3 && total > 0)
        .filter(|(word, _)| word.len() > 4)
        .map(|(word, count)| (word.clone(), *count as f64 / total as f64))
        .filter(|(_, f)| *f > 0.001)
        .collect();

    VocabularyAnalysis {
        vocabulary_size: user_vocab.len() as u64,
        banned_words: banned,
        preferred_words: preferred,
    }
}

pub struct VocabularyAnalysis {
    pub vocabulary_size: u64,
    pub banned_words: Vec<String>,
    pub preferred_words: Vec<(String, f64)>,
}
