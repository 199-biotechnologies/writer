//! Inference backends (Ollama, mistral.rs, custom MLX, ...).
//!
//! Every backend implements [`InferenceBackend`]. The trait and its
//! companion types are defined in child modules.

pub mod capabilities;
pub mod request;
pub mod response;
