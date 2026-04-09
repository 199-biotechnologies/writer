//! Character n-gram frequency profiles.
use std::collections::HashMap;

pub fn compute_ngrams(text: &str, n: usize, top_k: usize) -> Vec<(String, u64)> {
    let text_lower = text.to_lowercase();
    let chars: Vec<char> = text_lower.chars().collect();

    if chars.len() < n {
        return Vec::new();
    }

    let mut counts: HashMap<String, u64> = HashMap::new();
    for window in chars.windows(n) {
        let gram: String = window.iter().collect();
        *counts.entry(gram).or_insert(0) += 1;
    }

    let mut sorted: Vec<(String, u64)> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    sorted.truncate(top_k);
    sorted
}

pub fn trigrams(text: &str) -> Vec<(String, u64)> {
    compute_ngrams(text, 3, 500)
}

pub fn quadgrams(text: &str) -> Vec<(String, u64)> {
    compute_ngrams(text, 4, 500)
}
