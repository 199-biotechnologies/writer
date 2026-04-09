//! Stylometric distance scoring.
//!
//! Computes a 0.0-1.0 distance between a text sample and a fingerprint.
//! Lower means closer to the author's voice.
use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

use crate::stylometry::ai_slop;
use crate::stylometry::features::function_words;
use crate::stylometry::features::lengths;
use crate::stylometry::features::ngrams;
use crate::stylometry::features::punctuation::PunctuationStats;
use crate::stylometry::fingerprint::StylometricFingerprint;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceReport {
    pub overall: f64,
    pub sentence_length_kl: f64,
    pub function_word_cos: f64,
    pub punctuation_l1: f64,
    pub ngram_cos: f64,
    pub ai_slop_penalty: f64,
}

/// Compute stylometric distance between text and a fingerprint.
/// Returns 0.0 (identical style) to 1.0 (maximally different).
pub fn distance(text: &str, fingerprint: &StylometricFingerprint) -> DistanceReport {
    let sentence_length_kl = sentence_length_divergence(text, fingerprint);
    let function_word_cos = function_word_distance(text, fingerprint);
    let punctuation_l1 = punctuation_distance(text, fingerprint);
    let ngram_cos = ngram_distance(text, fingerprint);
    let ai_slop_penalty = slop_penalty(text);

    // Weighted combination
    let overall = (sentence_length_kl * 0.25
        + function_word_cos * 0.25
        + punctuation_l1 * 0.15
        + ngram_cos * 0.20
        + ai_slop_penalty * 0.15)
        .clamp(0.0, 1.0);

    DistanceReport {
        overall,
        sentence_length_kl,
        function_word_cos,
        punctuation_l1,
        ngram_cos,
        ai_slop_penalty,
    }
}

fn sentence_length_divergence(text: &str, fp: &StylometricFingerprint) -> f64 {
    let text_stats = lengths::sentence_lengths(text);
    if fp.sentence_length.mean == 0.0 || text_stats.mean == 0.0 {
        return 0.5;
    }

    // Simplified KL-like divergence using mean and SD
    let mean_diff = ((text_stats.mean - fp.sentence_length.mean) / fp.sentence_length.mean.max(1.0)).abs();
    let sd_diff = if fp.sentence_length.sd > 0.0 {
        ((text_stats.sd - fp.sentence_length.sd) / fp.sentence_length.sd).abs()
    } else {
        0.0
    };

    ((mean_diff + sd_diff) / 2.0).clamp(0.0, 1.0)
}

fn function_word_distance(text: &str, fp: &StylometricFingerprint) -> f64 {
    let text_fw = function_words::compute(text);

    if fp.function_words.is_empty() || text_fw.is_empty() {
        return 0.5;
    }

    // Cosine distance between function word frequency vectors
    let all_keys: Vec<&String> = fp
        .function_words
        .keys()
        .chain(text_fw.keys())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for key in &all_keys {
        let a = fp.function_words.get(*key).copied().unwrap_or(0.0);
        let b = text_fw.get(*key).copied().unwrap_or(0.0);
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return 0.5;
    }

    let cosine_sim = dot / denom;
    (1.0 - cosine_sim).clamp(0.0, 1.0)
}

fn punctuation_distance(text: &str, fp: &StylometricFingerprint) -> f64 {
    let text_punct = PunctuationStats::compute(text);
    let fp_p = &fp.punctuation;

    // L1 distance normalized by typical ranges
    let diffs = [
        (text_punct.em_dashes_per_1k - fp_p.em_dashes_per_1k).abs() / 20.0,
        (text_punct.en_dashes_per_1k - fp_p.en_dashes_per_1k).abs() / 20.0,
        (text_punct.semicolons_per_1k - fp_p.semicolons_per_1k).abs() / 10.0,
        (text_punct.colons_per_1k - fp_p.colons_per_1k).abs() / 10.0,
        (text_punct.exclamations_per_1k - fp_p.exclamations_per_1k).abs() / 10.0,
        (text_punct.questions_per_1k - fp_p.questions_per_1k).abs() / 10.0,
        (text_punct.parentheses_per_1k - fp_p.parentheses_per_1k).abs() / 20.0,
    ];

    let avg: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;
    avg.clamp(0.0, 1.0)
}

fn ngram_distance(text: &str, fp: &StylometricFingerprint) -> f64 {
    let text_ngrams = ngrams::trigrams(text);

    if fp.ngram_profile.is_empty() || text_ngrams.is_empty() {
        return 0.5;
    }

    // Convert to frequency maps for cosine distance
    let fp_total: u64 = fp.ngram_profile.iter().map(|(_, c)| c).sum();
    let text_total: u64 = text_ngrams.iter().map(|(_, c)| c).sum();

    if fp_total == 0 || text_total == 0 {
        return 0.5;
    }

    let fp_map: std::collections::HashMap<&str, f64> = fp
        .ngram_profile
        .iter()
        .map(|(g, c)| (g.as_str(), *c as f64 / fp_total as f64))
        .collect();

    let text_map: std::collections::HashMap<&str, f64> = text_ngrams
        .iter()
        .map(|(g, c)| (g.as_str(), *c as f64 / text_total as f64))
        .collect();

    let all_keys: std::collections::HashSet<&str> = fp_map
        .keys()
        .chain(text_map.keys())
        .copied()
        .collect();

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for key in &all_keys {
        let a = fp_map.get(key).copied().unwrap_or(0.0);
        let b = text_map.get(key).copied().unwrap_or(0.0);
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return 0.5;
    }

    (1.0 - dot / denom).clamp(0.0, 1.0)
}

fn slop_penalty(text: &str) -> f64 {
    let text_lower = text.to_lowercase();
    let word_count = text.unicode_words().count() as f64;
    if word_count == 0.0 {
        return 0.0;
    }

    let word_hits: usize = ai_slop::BANNED_WORDS
        .iter()
        .filter(|w| text_lower.contains(**w))
        .count();

    let phrase_hits: usize = ai_slop::BANNED_PHRASES
        .iter()
        .filter(|p| text_lower.contains(**p))
        .count();

    let total_hits = word_hits + phrase_hits * 2; // phrases count double
    let density = total_hits as f64 / word_count * 100.0;

    // Scale: 0 hits = 0.0, 5+ per 100 words = 1.0
    (density / 5.0).clamp(0.0, 1.0)
}
