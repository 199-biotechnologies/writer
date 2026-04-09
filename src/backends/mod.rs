//! Pluggable inference and training backends.
//!
//! Writer never calls an external runtime directly. All inference goes
//! through [`inference::InferenceBackend`] and all training goes through
//! [`training::TrainingBackend`]. Swapping Ollama for mistral.rs, adding
//! TurboQuant, or wiring up a new quantisation scheme is a change behind
//! the trait, never in the commands or the decoding layer.

pub mod inference;
pub mod training;
pub mod types;
