//! Types shared across inference and training backends.
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// A globally-addressable model identifier in `owner/name` form.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId {
    owner: String,
    name: String,
}

impl ModelId {
    pub fn new(owner: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            owner: owner.into(),
            name: name.into(),
        }
    }

    pub fn owner(&self) -> &str {
        &self.owner
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.owner, self.name)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ModelIdParseError {
    #[error("expected `owner/name`, got `{0}`")]
    MissingOwner(String),
}

impl FromStr for ModelId {
    type Err = ModelIdParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (owner, name) = s
            .split_once('/')
            .ok_or_else(|| ModelIdParseError::MissingOwner(s.to_string()))?;
        Ok(Self {
            owner: owner.to_string(),
            name: name.to_string(),
        })
    }
}

/// Opaque handle returned by a backend after a model is loaded.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelHandle(pub String);

/// Reference to a LoRA adapter on disk, associated with a profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterRef {
    pub profile: String,
    pub path: PathBuf,
}

impl AdapterRef {
    pub fn new(profile: impl Into<String>, path: PathBuf) -> Self {
        Self {
            profile: profile.into(),
            path,
        }
    }
}
