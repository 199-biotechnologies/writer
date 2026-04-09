//! Streaming response vocabulary.
//!
//! Backends stream `GenerationEvent`s back to the decoding layer. Events
//! are tagged with a `candidate_index` so a single stream can carry N
//! parallel candidates (needed for generate-N-rank).
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GenerationEvent {
    Token {
        candidate_index: u16,
        text: String,
        logprob: f32,
    },
    Done {
        candidate_index: u16,
        finish_reason: FinishReason,
        usage: UsageStats,
        full_text: String,
    },
    Error {
        candidate_index: u16,
        message: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    Stop,
    MaxTokens,
    FilteredByBackend,
    Aborted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageStats {
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    pub elapsed_ms: u64,
}
