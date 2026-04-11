//! Vocabulary richness metrics — evidence-based authorship signals.
//!
//! References:
//! - Yule, G.U. (1944). "The Statistical Study of Literary Vocabulary." Cambridge.
//! - Simpson, E.H. (1949). "Measurement of Diversity." Nature.
//! - Tweedie, F.J. & Baayen, R.H. (1998). "How Variable May a Constant Be?"
//!   Computers and the Humanities.
//! - Writeprints (Abbasi & Chen, 2008) uses hapax legomena as a key feature.
use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RichnessStats {
    /// Type-Token Ratio (vocabulary / total words)
    pub ttr: f64,
    /// Yule's K — length-independent vocabulary richness measure
    /// (lower = richer vocabulary). Typical range: 50-300.
    pub yules_k: f64,
    /// Hapax legomena ratio — words appearing exactly once / total vocabulary
    /// Strong authorship signal per Writeprints (Abbasi & Chen, 2008).
    pub hapax_legomena_ratio: f64,
    /// Hapax dislegomena ratio — words appearing exactly twice / total vocabulary
    pub hapax_dislegomena_ratio: f64,
    /// Simpson's Diversity Index (probability that two randomly chosen words differ)
    pub simpsons_d: f64,
}

impl RichnessStats {
    pub fn compute(text: &str) -> Self {
        let words: Vec<String> = text.unicode_words().map(|w| w.to_lowercase()).collect();

        let n = words.len() as f64;
        if n == 0.0 {
            return Self::default();
        }

        let mut freq: HashMap<&str, usize> = HashMap::new();
        for word in &words {
            *freq.entry(word.as_str()).or_insert(0) += 1;
        }

        let v = freq.len() as f64; // vocabulary size (types)
        let ttr = v / n;

        // Frequency spectrum: how many words appear exactly i times
        let mut spectrum: HashMap<usize, usize> = HashMap::new();
        for &count in freq.values() {
            *spectrum.entry(count).or_insert(0) += 1;
        }

        let hapax = *spectrum.get(&1).unwrap_or(&0) as f64;
        let hapax_dis = *spectrum.get(&2).unwrap_or(&0) as f64;

        let hapax_legomena_ratio = if v > 0.0 { hapax / v } else { 0.0 };
        let hapax_dislegomena_ratio = if v > 0.0 { hapax_dis / v } else { 0.0 };

        // Yule's K = 10^4 * (M2 - N) / N^2
        // where M2 = sum(i^2 * V_i) and V_i = number of types with frequency i
        let m2: f64 = spectrum
            .iter()
            .map(|(&i, &vi)| (i as f64).powi(2) * vi as f64)
            .sum();
        let yules_k = if n > 1.0 {
            10_000.0 * (m2 - n) / (n * n)
        } else {
            0.0
        };

        // Simpson's D = 1 - sum(n_i * (n_i - 1)) / (N * (N - 1))
        let sum_ni: f64 = freq.values().map(|&ni| ni as f64 * (ni as f64 - 1.0)).sum();
        let simpsons_d = if n > 1.0 {
            1.0 - sum_ni / (n * (n - 1.0))
        } else {
            0.0
        };

        Self {
            ttr,
            yules_k,
            hapax_legomena_ratio,
            hapax_dislegomena_ratio,
            simpsons_d,
        }
    }
}
