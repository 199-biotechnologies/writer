//! Function word frequency — a strong stylometric signal.
use std::collections::HashMap;

use unicode_segmentation::UnicodeSegmentation;

const FUNCTION_WORDS: &[&str] = &[
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what", "so",
    "up", "out", "if", "about", "who", "get", "which", "go", "me", "when",
    "make", "can", "like", "no", "just", "him", "know", "take", "could", "them",
    "than", "other", "been", "now", "its", "also", "after", "very", "then", "our",
    "into", "well", "some", "had", "only", "how", "has", "may", "more", "over",
    "such", "any", "most", "each", "much", "both", "where", "being", "while", "through",
    "before", "between", "own", "these", "those", "even", "because", "way", "still", "here",
    "should", "many", "must", "under", "since", "might", "yet", "us", "same", "though",
    "did", "too", "every", "until", "around", "without", "again", "why", "few", "never",
    "were", "was", "are", "is", "am", "been", "done", "does", "got", "shall",
    "during", "another", "off", "down", "let", "once", "already", "almost", "always",
    "nor", "either", "neither", "whether", "nothing", "himself", "herself", "itself",
    "themselves", "something", "anything", "everything", "someone", "anyone", "everyone",
    "whom", "whose", "upon", "thus", "hence", "therefore", "however", "although",
    "moreover", "furthermore", "nevertheless", "whereas", "whereby", "wherein",
    "perhaps", "indeed", "rather", "quite", "somewhat", "enough", "within", "among",
    "against", "above", "below", "beside", "beyond", "throughout", "toward", "towards",
    "onto", "except", "along", "across", "behind", "beneath",
];

pub fn compute(text: &str) -> HashMap<String, f64> {
    let words: Vec<String> = text
        .unicode_words()
        .map(|w| w.to_lowercase())
        .collect();

    let total = words.len() as f64;
    if total == 0.0 {
        return HashMap::new();
    }

    let mut counts: HashMap<String, usize> = HashMap::new();
    for word in &words {
        if FUNCTION_WORDS.contains(&word.as_str()) {
            *counts.entry(word.clone()).or_insert(0) += 1;
        }
    }

    counts
        .into_iter()
        .map(|(word, count)| (word, count as f64 / total))
        .collect()
}
