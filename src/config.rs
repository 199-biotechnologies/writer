/// Configuration loading with 3-tier precedence:
///   1. Compiled defaults
///   2. TOML config file (~/.config/writer/config.toml)
///   3. Environment variables (WRITER_*)
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::error::AppError;

// ── Config structs ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Name of the currently active voice profile
    pub active_profile: String,

    /// Base model to use for training and inference
    pub base_model: String,

    /// Self-update settings
    pub update: UpdateConfig,

    /// Inference settings
    pub inference: InferenceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateConfig {
    pub enabled: bool,
    pub owner: String,
    pub repo: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Sampling temperature (0.0 - 2.0)
    pub temperature: f32,
    /// Max new tokens per generation
    pub max_tokens: u32,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            active_profile: "default".into(),
            base_model: "llama-3.2-3b-instruct".into(),
            update: UpdateConfig::default(),
            inference: InferenceConfig::default(),
        }
    }
}

impl Default for UpdateConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            owner: "199-biotechnologies".into(),
            repo: "writer".into(),
        }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            max_tokens: 1024,
        }
    }
}

// ── Paths ──────────────────────────────────────────────────────────────────

pub fn config_dir() -> PathBuf {
    directories::ProjectDirs::from("", "", "writer")
        .map(|d| d.config_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn data_dir() -> PathBuf {
    directories::ProjectDirs::from("", "", "writer")
        .map(|d| d.data_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn config_path() -> PathBuf {
    config_dir().join("config.toml")
}

pub fn profiles_dir() -> PathBuf {
    data_dir().join("profiles")
}

pub fn models_dir() -> PathBuf {
    data_dir().join("models")
}

// ── Loading ────────────────────────────────────────────────────────────────

pub fn load() -> Result<AppConfig, AppError> {
    use figment::Figment;
    use figment::providers::{Env, Format as _, Serialized, Toml};

    Figment::from(Serialized::defaults(AppConfig::default()))
        .merge(Toml::file(config_path()))
        .merge(Env::prefixed("WRITER_").split("_"))
        .extract()
        .map_err(|e| AppError::Config(e.to_string()))
}
