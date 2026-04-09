# writer Quality Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build writer into a local-first CLI whose generated text is indistinguishable from the user's own writing, with an architecture that lets us drop in TurboQuant, TriAttention, speculative decoding, and future optimisations as they land without rewriting the core.

**Architecture:** Trait-based backends (inference + training) behind thin HTTP/subprocess adapters. Data pipeline composes source parsers, normalisation, and stylometric fingerprinting into self-contained artefacts. A decoding layer wraps every inference call with logit bias, generate-N-rank, contrastive decoding, and post-hoc filtering. Every layer is a pure-Rust module; every external dependency (Ollama, mlx-tune, Hugging Face) is behind a trait so we can swap implementations when something faster arrives.

**Tech Stack:**
- **Language/edition:** Rust 2024, MSRV 1.85
- **CLI:** clap 4.5 derive
- **Async:** tokio 1.x with `rt-multi-thread`, `macros`, `time`
- **HTTP:** reqwest 0.12 with `json`, `stream`, `rustls-tls`
- **Serde:** serde 1.x, serde_json 1.x
- **Config:** figment 0.10 (toml + env)
- **Errors:** thiserror 2.x
- **Paths:** directories 6.x
- **Tables/colour:** comfy-table 7, owo-colors 4
- **Self-update:** self_update 0.42
- **Text processing:** unicode-segmentation 1.x, regex 1.x
- **Archives:** zip 2.x (for Twitter/LinkedIn archives)
- **HTML:** scraper 0.20 (for URL ingestion)
- **Quantiles / stats:** hdrhistogram 7.x (for rhythm distributions)
- **Safetensors:** safetensors 0.4 (adapter file parsing)
- **Dev:** assert_cmd 2, predicates 3, tempfile 3, wiremock 0.6 (mock Ollama HTTP)
- **External tools (shell out, not linked):** Ollama ≥ 0.19 (MLX-powered), mlx-tune ≥ 0.3 for training on Apple Silicon, unsloth for NVIDIA (optional)

**Non-negotiables (quality bar, never compromised):**
1. **Every quality layer is opt-out, not opt-in.** The default `writer write` runs the full stack: fingerprint-biased decoding, generate-N-rank, post-hoc filtering, contrastive decoding against the base model. Speed comes from MoE and caching, not from skipping quality layers.
2. **The user's data never leaves the machine.** No telemetry. No cloud API calls. Any network call in the hot path is a bug.
3. **Voice fidelity is measurable.** `writer score` produces a single stylometric distance number. Every change to the decoding stack must be benchmarked against the user's own held-out samples and must not regress fidelity.
4. **Extensibility is structural, not rhetorical.** TurboQuant and TriAttention have to be addable as backend features in under a day, with zero changes to `src/commands/`, the data pipeline, or the decoding layer. If a hypothetical TurboQuant integration would touch any of those, the design is wrong.
5. **No silent fallbacks.** When a layer cannot run (backend does not support contrastive decoding, training data is too small, adapter missing), writer returns a clear structured error. It does not quietly use a worse path.
6. **One static binary.** Writer ships as a single Rust binary. External tools (Ollama, mlx-tune) are detected, never bundled, never auto-installed. `writer init` produces a diagnostic report and instructions if a tool is missing.

---

## Architecture Overview

### Module layout (target after Phase 2)

```
src/
├── main.rs                    # entry: parse, detect format, dispatch, exit
├── cli.rs                     # clap derive: Cli + Commands + Args
├── config.rs                  # AppConfig + figment loading + paths
├── error.rs                   # AppError enum (unchanged from v0.1)
├── output.rs                  # JSON envelope, Format, Ctx (unchanged)
│
├── backends/                  # pluggable inference + training
│   ├── mod.rs                 # re-exports
│   ├── types.rs               # shared: ModelId, ModelHandle, AdapterRef
│   ├── inference/
│   │   ├── mod.rs             # InferenceBackend trait + registry
│   │   ├── capabilities.rs    # BackendCapabilities struct
│   │   ├── request.rs         # GenerationRequest, GenerationParams
│   │   ├── response.rs        # GenerationEvent, StreamingResponse
│   │   └── ollama.rs          # Ollama HTTP backend implementation
│   └── training/
│       ├── mod.rs             # TrainingBackend trait + registry
│       ├── config.rs          # LoraConfig, DpoConfig, TrainingProgress
│       ├── artefact.rs        # AdapterArtifact (path, metadata)
│       └── mlx_tune.rs        # mlx-tune subprocess backend
│
├── corpus/                    # data pipeline: sources -> samples
│   ├── mod.rs
│   ├── sample.rs              # Sample struct: content, metadata, hash
│   ├── sources/
│   │   ├── mod.rs             # Source trait + registry
│   │   ├── markdown.rs
│   │   ├── plain_text.rs
│   │   ├── twitter_archive.rs
│   │   ├── obsidian.rs
│   │   └── url.rs
│   ├── normalize.rs           # strip signatures, YAML, mentions, tracking links
│   ├── dedupe.rs              # content-hash dedupe
│   └── chunker.rs             # split long docs for training window
│
├── stylometry/                # fingerprint computation + scoring
│   ├── mod.rs
│   ├── fingerprint.rs         # StylometricFingerprint struct + compute()
│   ├── features/
│   │   ├── mod.rs
│   │   ├── lengths.rs         # word/sentence/paragraph length distributions
│   │   ├── function_words.rs  # stopword frequency (stylometric signal)
│   │   ├── ngrams.rs          # char n-gram profile
│   │   ├── punctuation.rs     # punctuation frequency
│   │   └── vocabulary.rs      # banned/preferred word lists per profile
│   ├── scoring.rs             # distance(text, fingerprint) -> f64
│   ├── dashboard.rs           # HTML dashboard export
│   └── ai_slop.rs             # embedded banned-word / banned-phrase lists
│
├── decoding/                  # quality wrapper around InferenceBackend
│   ├── mod.rs                 # run(request) -> samples through the whole stack
│   ├── logit_bias.rs          # fingerprint -> logit bias map
│   ├── contrastive.rs         # CoPe-style contrastive decoding glue
│   ├── ranker.rs              # generate-N-rank by stylometric distance
│   ├── filter.rs              # post-hoc structural filtering
│   └── prompts.rs             # base/write/rewrite prompt templates
│
├── profile/                   # profile directory management
│   ├── mod.rs
│   ├── store.rs               # on-disk operations on profile dirs
│   └── state.rs               # active profile tracking
│
└── commands/                  # thin dispatchers
    ├── mod.rs
    ├── init.rs                # bootstrap directories + diagnostics
    ├── learn.rs               # corpus ingestion entry point
    ├── profile.rs             # show/list/new/use
    ├── train.rs               # backends::training orchestration
    ├── write.rs               # decoding::run entry point
    ├── rewrite.rs              # decoding::run with rewrite template
    ├── score.rs               # stylometric distance scoring (NEW)
    ├── model.rs               # backends::inference model management
    ├── agent_info.rs          # capability manifest (updated)
    ├── skill.rs               # SKILL.md install (unchanged)
    ├── config.rs              # show/path (unchanged)
    └── update.rs              # self-update (unchanged)
```

### Data flow: `writer write "prompt"` (the critical path)

```
writer write "essay about X"
    │
    ▼
commands::write::run(ctx, prompt)
    │
    ▼
decoding::run(prompt, profile, config)
    │
    ├─ load fingerprint from profile
    ├─ build GenerationRequest
    │      prompt_template: write::write()
    │      n_candidates: config.decoding.n_candidates (default 8)
    │      logit_bias: fingerprint.banned_word_bias() + ai_slop::default_bias()
    │      contrastive_base: config.decoding.contrastive_base (optional)
    │      max_tokens, temperature, top_p from config
    │
    ▼
InferenceBackend::generate(request)              // Ollama or future backend
    │
    ├─ loads model + adapter (if active profile trained)
    ├─ applies logit bias
    ├─ optional: contrastive decoding (dual forward pass)
    ├─ optional: KV cache quant (TurboQuant, TriAttention -- future)
    ├─ optional: speculative decoding (draft model -- future)
    │
    ▼
returns Vec<Candidate>
    │
    ▼
decoding::ranker::rank(candidates, fingerprint)
    │
    ▼
decoding::filter::check(best, fingerprint)
    │
    ├─ if passes: return
    ├─ if fails: regenerate up to 3 times, fail loud
    │
    ▼
output::print_success_or(ctx, result, human_formatter)
```

Every arrow in that diagram crosses a trait boundary or a module boundary. TurboQuant is added by flipping a capability flag on an existing backend, or by introducing a new backend. Nothing above the `InferenceBackend::generate` line changes.

### Data flow: `writer learn <files>` + `writer train`

```
writer learn ~/twitter.zip
    │
    ▼
commands::learn::run(ctx, files)
    │
    ▼
corpus::ingest(files, config)
    │
    ├─ for each file/dir/URL:
    │     source = sources::detect(path)
    │     samples = source.parse(path)
    │     samples = normalize::clean(samples)
    │     samples = dedupe::run(samples, profile)
    │     samples = chunker::split(samples, config.train.max_seq_len)
    │
    ▼
Vec<Sample> written to profile/<name>/samples/*.jsonl
stylometry::fingerprint::compute(samples) -> profile/<name>/fingerprint.json

writer train
    │
    ▼
commands::train::run(ctx, profile)
    │
    ▼
backends::training::mlx_tune::train_lora(LoraConfig{
    model: config.base_model,
    dataset: profile/<name>/samples/,
    adapter_out: profile/<name>/adapter.safetensors,
    ...
})
    │
    ▼
spawn("mlx_lm.lora --train --model gemma-4-26b-a4b ...")
    ├─ stream stdout/stderr to ctx
    ├─ parse progress JSON lines
    │
    ▼
AdapterArtifact { path, metadata, loss_curve }
    │
    ▼
profile::store::attach_adapter(profile, artefact)
```

Notice training never touches `InferenceBackend`. The backend abstraction is strictly between inference and the decoding layer. Training writes an adapter file to disk; inference reads it. No cross-talk.

### Extension point: adding TurboQuant (worked example)

TurboQuant compresses the KV cache during inference. It is an inference-time optimisation. When it ships in Ollama (likely weeks):

1. Add a field to `BackendCapabilities`:
   ```rust
   pub struct BackendCapabilities {
       pub kv_quant: Option<KvQuantKind>,  // None, TurboQuant, TriAttention, Both
       // ...
   }
   ```
2. Add a field to `GenerationParams`:
   ```rust
   pub struct GenerationParams {
       pub kv_quant: KvQuantConfig,  // defaults to `Auto` = use whatever backend advertises
       // ...
   }
   ```
3. In `backends::inference::ollama::OllamaBackend::generate`, set the relevant Ollama options (`options.kv_quant_type = "turboquant_q3"` or whatever Ollama names it).
4. Update the `writer config set decoding.kv_quant <kind>` command to accept the new value.
5. Add a test that asserts the backend advertises the capability when the installed Ollama version supports it.

That is the entire change. Zero lines touched in `src/commands/`, `src/corpus/`, `src/decoding/`, or `src/stylometry/`. The extension point exists because inference parameters are a typed struct, not string-interpolated CLI flags, and because the backend reports capabilities so the decoding layer can negotiate.

Same pattern applies to TriAttention (a second KV optimisation flag), speculative decoding (a `draft_model: Option<ModelId>` field on `GenerationParams`), and future techniques we have not heard of yet.

---

## Phases

The plan has ten phases. Phase 0 is a pure refactor that unlocks everything else and has full TDD detail below. Phases 1-3 have task-level specifications with file paths, interfaces, and acceptance criteria — detailed enough to execute. Phases 4-9 have module layouts and behaviour contracts — they will be expanded into full TDD task lists when we reach them.

| Phase | Scope | Status | Detail level |
|---|---|---|---|
| 0 | Foundation: traits, types, backend registry | Blocking everything | **Full TDD bite-sized tasks** |
| 1 | Data pipeline: sources, normalise, chunk, dedupe | Quality starts here | **Task specifications** |
| 2 | Stylometric fingerprint v2 + HTML dashboard | Voice measurement | **Task specifications** |
| 3 | Ollama inference backend | Make `writer write` real | **Task specifications** |
| 4 | Quality decoding: logit bias, generate-N-rank, filters | Quality layer on top of inference | Module + contract |
| 5 | mlx-tune training backend: LoRA fine-tune | Learn the voice | Module + contract |
| 6 | DPO against AI-rewrites | Teach the model what NOT to do | Module + contract |
| 7 | Contrastive decoding (CoPe) | Subtract the base distribution at decode time | Module + contract |
| 8 | `writer score` stylometric distance | Quality feedback loop | Module + contract |
| 9 | Extensibility reference implementation + docs | Prove the architecture | Module + contract |

---

## Phase 0 — Foundation (full TDD)

**Goal:** introduce the trait-based backend architecture and the core type vocabulary (Sample, StylometricFingerprint, GenerationRequest, LoraConfig) without changing any command behaviour. At the end of Phase 0, `writer --help`, `writer agent-info`, `writer init`, `writer learn`, and `writer profile show` work exactly as today. The difference is that the types exist and the traits are in place, ready for Phase 1 to plug in real implementations.

**Files created this phase:**
- `src/backends/mod.rs`
- `src/backends/types.rs`
- `src/backends/inference/mod.rs`
- `src/backends/inference/capabilities.rs`
- `src/backends/inference/request.rs`
- `src/backends/inference/response.rs`
- `src/backends/training/mod.rs`
- `src/backends/training/config.rs`
- `src/backends/training/artefact.rs`
- `src/corpus/mod.rs`
- `src/corpus/sample.rs`
- `src/stylometry/mod.rs`
- `src/stylometry/fingerprint.rs` (skeleton, real compute in Phase 2)
- `tests/phase0_types.rs`

**Files modified:**
- `src/main.rs` — add `mod backends;`, `mod corpus;`, `mod stylometry;`
- `Cargo.toml` — add `async-trait`, `hdrhistogram`, `unicode-segmentation`

### Task 0.1: add new crate dependencies

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1: add dependencies to Cargo.toml**

Add under `[dependencies]`:

```toml
async-trait = "0.1"
tokio = { version = "1", features = ["rt-multi-thread", "macros", "time", "fs", "process"] }
reqwest = { version = "0.12", features = ["json", "stream", "rustls-tls"], default-features = false }
hdrhistogram = "7"
unicode-segmentation = "1"
regex = "1"
```

Add under `[dev-dependencies]`:

```toml
wiremock = "0.6"
tokio-test = "0.4"
```

- [ ] **Step 2: verify the project still builds**

Run: `cargo build 2>&1 | tail -5`
Expected: `Finished` with no errors. Warnings about unused deps are fine at this stage.

- [ ] **Step 3: commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "build: add async-trait, tokio-runtime, reqwest, text deps"
```

### Task 0.2: create `backends/types.rs` — shared backend types

**Files:**
- Create: `src/backends/mod.rs`
- Create: `src/backends/types.rs`
- Create: `tests/phase0_types.rs`

- [ ] **Step 1: write the failing test**

Create `tests/phase0_types.rs`:

```rust
use writer_cli::backends::types::{ModelId, ModelHandle, AdapterRef};

#[test]
fn model_id_parses_owner_repo_format() {
    let id: ModelId = "google/gemma-4-26b-a4b".parse().unwrap();
    assert_eq!(id.owner(), "google");
    assert_eq!(id.name(), "gemma-4-26b-a4b");
    assert_eq!(id.to_string(), "google/gemma-4-26b-a4b");
}

#[test]
fn model_id_rejects_missing_owner() {
    let result: Result<ModelId, _> = "gemma-4-26b-a4b".parse();
    assert!(result.is_err());
}

#[test]
fn adapter_ref_carries_path_and_profile_name() {
    let r = AdapterRef::new("default", std::path::PathBuf::from("/tmp/a.safetensors"));
    assert_eq!(r.profile, "default");
    assert_eq!(r.path, std::path::PathBuf::from("/tmp/a.safetensors"));
}
```

- [ ] **Step 2: expose the crate as a library**

Modify `Cargo.toml` to add a `[lib]` section alongside the existing `[[bin]]`:

```toml
[lib]
name = "writer_cli"
path = "src/lib.rs"
```

Create `src/lib.rs`:

```rust
//! writer-cli library surface — used by integration tests.

pub mod backends;
pub mod corpus;
pub mod stylometry;
```

Modify `src/main.rs` to use the library module paths. Replace the top-of-file `mod backends;` etc. with:

```rust
use writer_cli::{backends, corpus, stylometry};
```

(Keep `mod cli; mod commands; mod config; mod error; mod output;` since those stay binary-private for now.)

- [ ] **Step 3: run the test to verify it fails**

Run: `cargo test --test phase0_types 2>&1 | tail -20`
Expected: `error[E0432]: unresolved import ...backends::types`

- [ ] **Step 4: create `src/backends/mod.rs`**

```rust
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
```

- [ ] **Step 5: create `src/backends/types.rs`**

```rust
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
        Self { owner: owner.into(), name: name.into() }
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
        Self { profile: profile.into(), path }
    }
}
```

- [ ] **Step 6: create empty `inference/mod.rs` and `training/mod.rs` stubs**

`src/backends/inference/mod.rs`:

```rust
//! Inference backends (Ollama, mistral.rs, custom MLX, ...).
//!
//! Every backend implements [`InferenceBackend`]. The trait and its
//! companion types are defined in child modules.

pub mod capabilities;
pub mod request;
pub mod response;
```

`src/backends/training/mod.rs`:

```rust
//! Training backends (mlx-tune, unsloth, ...).
//!
//! Every backend implements [`TrainingBackend`]. The trait and its
//! companion types are defined in child modules.

pub mod artefact;
pub mod config;
```

Create empty child module files to satisfy the module declarations:

`src/backends/inference/capabilities.rs`: `// placeholder — filled in Task 0.3`
`src/backends/inference/request.rs`: `// placeholder — filled in Task 0.4`
`src/backends/inference/response.rs`: `// placeholder — filled in Task 0.5`
`src/backends/training/artefact.rs`: `// placeholder — filled in Task 0.7`
`src/backends/training/config.rs`: `// placeholder — filled in Task 0.7`

- [ ] **Step 7: run the test to verify it passes**

Run: `cargo test --test phase0_types 2>&1 | tail -10`
Expected: `test result: ok. 3 passed`

- [ ] **Step 8: commit**

```bash
git add src/lib.rs src/main.rs src/backends/ tests/phase0_types.rs Cargo.toml
git commit -m "feat(backends): scaffold module tree + ModelId/ModelHandle/AdapterRef types"
```

### Task 0.3: `BackendCapabilities` struct

**Files:**
- Modify: `src/backends/inference/capabilities.rs`
- Modify: `tests/phase0_types.rs`

- [ ] **Step 1: extend the test file**

Append to `tests/phase0_types.rs`:

```rust
use writer_cli::backends::inference::capabilities::{
    BackendCapabilities, KvQuantKind, QuantSchemeKind,
};

#[test]
fn capabilities_default_is_the_narrowest_possible_backend() {
    let caps = BackendCapabilities::default();
    assert!(!caps.supports_lora);
    assert!(!caps.supports_logit_bias);
    assert!(!caps.supports_contrastive_decoding);
    assert!(!caps.supports_speculative_decoding);
    assert_eq!(caps.kv_quant, KvQuantKind::None);
    assert_eq!(caps.quant_schemes, vec![]);
    assert_eq!(caps.max_context, 2048);
}

#[test]
fn capabilities_builder_sets_flags() {
    let caps = BackendCapabilities {
        supports_lora: true,
        supports_logit_bias: true,
        kv_quant: KvQuantKind::TurboQuant,
        quant_schemes: vec![QuantSchemeKind::Q4KM, QuantSchemeKind::Q5KM],
        max_context: 128_000,
        ..Default::default()
    };
    assert!(caps.supports_lora);
    assert!(caps.supports_logit_bias);
    assert_eq!(caps.kv_quant, KvQuantKind::TurboQuant);
    assert_eq!(caps.quant_schemes.len(), 2);
    assert_eq!(caps.max_context, 128_000);
}
```

- [ ] **Step 2: run the test to verify it fails**

Run: `cargo test --test phase0_types 2>&1 | tail -10`
Expected: `error[E0432]: unresolved import ...backends::inference::capabilities::BackendCapabilities`

- [ ] **Step 3: implement `capabilities.rs`**

Replace the contents of `src/backends/inference/capabilities.rs`:

```rust
//! Capability advertisement for inference backends.
//!
//! Every backend returns a `BackendCapabilities` from
//! [`InferenceBackend::capabilities`]. The decoding layer inspects this
//! struct before calling `generate` to negotiate which quality features
//! are available. Flags default to off so unknown backends behave
//! conservatively.
use serde::{Deserialize, Serialize};

/// KV cache optimisation family supported by the backend.
///
/// Multiple kinds can stack (e.g. a backend may advertise `Both` if it
/// can run TurboQuant compression _and_ TriAttention pruning together).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum KvQuantKind {
    #[default]
    None,
    /// Google TurboQuant / PolarQuant / QJL KV-value quantisation.
    TurboQuant,
    /// MLX TriAttention key-pruning.
    TriAttention,
    /// Both families active simultaneously.
    Both,
}

/// Weight quantisation scheme the backend can load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum QuantSchemeKind {
    Bf16,
    Fp16,
    Q8_0,
    Q6K,
    Q5KM,
    Q4KM,
    Q3KM,
    AWQ,
    UnslothDynamic20,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendCapabilities {
    pub supports_lora: bool,
    pub supports_logit_bias: bool,
    pub supports_contrastive_decoding: bool,
    pub supports_speculative_decoding: bool,
    pub supports_activation_steering: bool,
    pub kv_quant: KvQuantKind,
    pub quant_schemes: Vec<QuantSchemeKind>,
    pub max_context: usize,
    pub streaming: bool,
}

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self {
            supports_lora: false,
            supports_logit_bias: false,
            supports_contrastive_decoding: false,
            supports_speculative_decoding: false,
            supports_activation_steering: false,
            kv_quant: KvQuantKind::None,
            quant_schemes: vec![],
            max_context: 2048,
            streaming: false,
        }
    }
}
```

- [ ] **Step 4: run the test**

Run: `cargo test --test phase0_types 2>&1 | tail -10`
Expected: `test result: ok. 5 passed`

- [ ] **Step 5: commit**

```bash
git add src/backends/inference/capabilities.rs tests/phase0_types.rs
git commit -m "feat(backends): BackendCapabilities with KvQuant/QuantScheme vocab"
```

### Task 0.4: `GenerationRequest` + `GenerationParams`

**Files:**
- Modify: `src/backends/inference/request.rs`
- Modify: `tests/phase0_types.rs`

- [ ] **Step 1: write the failing test**

Append to `tests/phase0_types.rs`:

```rust
use std::collections::HashMap;
use writer_cli::backends::inference::request::{
    GenerationRequest, GenerationParams, LogitBiasMap,
};
use writer_cli::backends::types::{AdapterRef, ModelId};

#[test]
fn generation_request_builder_defaults_are_quality_oriented() {
    let req = GenerationRequest::new(
        "google/gemma-4-26b-a4b".parse::<ModelId>().unwrap(),
        "write an essay about ravens".to_string(),
    );
    assert_eq!(req.params.n_candidates, 8);
    assert_eq!(req.params.temperature, 0.7);
    assert_eq!(req.params.top_p, 0.92);
    assert_eq!(req.params.max_tokens, 2048);
    assert_eq!(req.params.kv_quant_preference, None);
    assert!(req.adapter.is_none());
    assert!(req.logit_bias.is_empty());
}

#[test]
fn generation_request_with_adapter_and_bias() {
    let mut bias: LogitBiasMap = HashMap::new();
    bias.insert("delve".into(), -5.0);
    bias.insert("leverage".into(), -3.0);

    let req = GenerationRequest::new(
        "google/gemma-4-26b-a4b".parse::<ModelId>().unwrap(),
        "draft a blog post".to_string(),
    )
    .with_adapter(AdapterRef::new("default", "/tmp/a.safetensors".into()))
    .with_logit_bias(bias)
    .with_n_candidates(16);

    assert!(req.adapter.is_some());
    assert_eq!(req.logit_bias.get("delve"), Some(&-5.0));
    assert_eq!(req.params.n_candidates, 16);
}
```

- [ ] **Step 2: run to verify it fails**

Run: `cargo test --test phase0_types 2>&1 | tail -15`
Expected: `error[E0432]: unresolved import ...request::GenerationRequest`

- [ ] **Step 3: implement `request.rs`**

Replace contents of `src/backends/inference/request.rs`:

```rust
//! Inference request vocabulary.
//!
//! `GenerationRequest` is the single struct every `InferenceBackend`
//! receives from the decoding layer. New features — contrastive
//! decoding, speculative decoding, KV quant preferences — are added as
//! new fields on `GenerationParams`, never as parallel code paths.
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::capabilities::KvQuantKind;
use crate::backends::types::{AdapterRef, ModelId};

pub type LogitBiasMap = HashMap<String, f32>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRequest {
    pub model: ModelId,
    pub prompt: String,
    pub adapter: Option<AdapterRef>,
    pub params: GenerationParams,
    pub logit_bias: LogitBiasMap,
    pub stop_sequences: Vec<String>,
    pub system_prompt: Option<String>,
    pub draft_model: Option<ModelId>,
    pub contrastive_base: Option<ModelId>,
}

impl GenerationRequest {
    pub fn new(model: ModelId, prompt: String) -> Self {
        Self {
            model,
            prompt,
            adapter: None,
            params: GenerationParams::default(),
            logit_bias: HashMap::new(),
            stop_sequences: Vec::new(),
            system_prompt: None,
            draft_model: None,
            contrastive_base: None,
        }
    }

    pub fn with_adapter(mut self, adapter: AdapterRef) -> Self {
        self.adapter = Some(adapter);
        self
    }

    pub fn with_logit_bias(mut self, bias: LogitBiasMap) -> Self {
        self.logit_bias = bias;
        self
    }

    pub fn with_n_candidates(mut self, n: u16) -> Self {
        self.params.n_candidates = n;
        self
    }

    pub fn with_contrastive_base(mut self, base: ModelId) -> Self {
        self.contrastive_base = Some(base);
        self
    }

    pub fn with_draft_model(mut self, draft: ModelId) -> Self {
        self.draft_model = Some(draft);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationParams {
    pub n_candidates: u16,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub max_tokens: u32,
    pub repetition_penalty: f32,
    pub contrastive_alpha: f32,
    pub kv_quant_preference: Option<KvQuantKind>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        // Quality-first defaults. The decoding layer ranks N candidates
        // and returns the best. Temperature and top_p match the setting
        // that produces the widest style variance without drifting from
        // the fine-tuned distribution.
        Self {
            n_candidates: 8,
            temperature: 0.7,
            top_p: 0.92,
            top_k: 64,
            max_tokens: 2048,
            repetition_penalty: 1.05,
            contrastive_alpha: 0.0,
            kv_quant_preference: None,
        }
    }
}
```

- [ ] **Step 4: run the test**

Run: `cargo test --test phase0_types 2>&1 | tail -10`
Expected: `test result: ok. 7 passed`

- [ ] **Step 5: commit**

```bash
git add src/backends/inference/request.rs tests/phase0_types.rs
git commit -m "feat(backends): GenerationRequest + GenerationParams with quality-first defaults"
```

### Task 0.5: `GenerationEvent` + `StreamingResponse`

**Files:**
- Modify: `src/backends/inference/response.rs`
- Modify: `tests/phase0_types.rs`

- [ ] **Step 1: write the failing test**

Append to `tests/phase0_types.rs`:

```rust
use writer_cli::backends::inference::response::{GenerationEvent, FinishReason, UsageStats};

#[test]
fn generation_event_token_carries_text_and_logprob() {
    let evt = GenerationEvent::Token {
        candidate_index: 0,
        text: "raven".into(),
        logprob: -1.2,
    };
    match evt {
        GenerationEvent::Token { candidate_index, text, logprob } => {
            assert_eq!(candidate_index, 0);
            assert_eq!(text, "raven");
            assert!((logprob + 1.2).abs() < 1e-6);
        }
        _ => panic!("expected Token"),
    }
}

#[test]
fn generation_event_done_has_finish_reason_and_usage() {
    let evt = GenerationEvent::Done {
        candidate_index: 0,
        finish_reason: FinishReason::Stop,
        usage: UsageStats { prompt_tokens: 42, generated_tokens: 168, elapsed_ms: 1234 },
        full_text: "raven are clever birds".into(),
    };
    if let GenerationEvent::Done { usage, finish_reason, .. } = evt {
        assert_eq!(finish_reason, FinishReason::Stop);
        assert_eq!(usage.prompt_tokens, 42);
        assert_eq!(usage.generated_tokens, 168);
    } else {
        panic!("expected Done");
    }
}
```

- [ ] **Step 2: run to verify it fails**

Run: `cargo test --test phase0_types 2>&1 | tail -10`
Expected: `error[E0432]: ...response::GenerationEvent`

- [ ] **Step 3: implement `response.rs`**

Replace contents of `src/backends/inference/response.rs`:

```rust
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
```

- [ ] **Step 4: run the test**

Run: `cargo test --test phase0_types 2>&1 | tail -10`
Expected: `test result: ok. 9 passed`

- [ ] **Step 5: commit**

```bash
git add src/backends/inference/response.rs tests/phase0_types.rs
git commit -m "feat(backends): GenerationEvent streaming + UsageStats"
```

### Task 0.6: `InferenceBackend` trait + no-op test backend

**Files:**
- Modify: `src/backends/inference/mod.rs`
- Create: `tests/phase0_inference_backend.rs`

- [ ] **Step 1: write the failing test**

Create `tests/phase0_inference_backend.rs`:

```rust
use async_trait::async_trait;
use tokio_stream::StreamExt;
use writer_cli::backends::inference::capabilities::BackendCapabilities;
use writer_cli::backends::inference::request::GenerationRequest;
use writer_cli::backends::inference::response::{FinishReason, GenerationEvent, UsageStats};
use writer_cli::backends::inference::{BackendError, InferenceBackend, ModelListing};
use writer_cli::backends::types::{ModelHandle, ModelId};

struct FakeBackend;

#[async_trait]
impl InferenceBackend for FakeBackend {
    fn name(&self) -> &str {
        "fake"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_lora: true,
            supports_logit_bias: true,
            max_context: 8192,
            streaming: true,
            ..Default::default()
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelListing>, BackendError> {
        Ok(vec![ModelListing {
            id: "google/gemma-4-26b-a4b".parse().unwrap(),
            is_downloaded: true,
            size_bytes: Some(14_000_000_000),
        }])
    }

    async fn load_model(&self, id: &ModelId) -> Result<ModelHandle, BackendError> {
        Ok(ModelHandle(format!("fake:{id}")))
    }

    async fn generate(
        &self,
        _handle: &ModelHandle,
        request: GenerationRequest,
    ) -> Result<Box<dyn tokio_stream::Stream<Item = GenerationEvent> + Send + Unpin>, BackendError> {
        let n = request.params.n_candidates as usize;
        let events: Vec<GenerationEvent> = (0..n)
            .flat_map(|i| {
                vec![
                    GenerationEvent::Token {
                        candidate_index: i as u16,
                        text: format!("candidate-{i}"),
                        logprob: -0.5,
                    },
                    GenerationEvent::Done {
                        candidate_index: i as u16,
                        finish_reason: FinishReason::Stop,
                        usage: UsageStats { prompt_tokens: 1, generated_tokens: 1, elapsed_ms: 1 },
                        full_text: format!("candidate-{i}"),
                    },
                ]
            })
            .collect();
        Ok(Box::new(tokio_stream::iter(events)))
    }
}

#[tokio::test]
async fn fake_backend_round_trip() {
    let backend = FakeBackend;
    assert_eq!(backend.name(), "fake");
    assert!(backend.capabilities().supports_lora);

    let models = backend.list_models().await.unwrap();
    assert_eq!(models.len(), 1);

    let handle = backend.load_model(&models[0].id).await.unwrap();
    assert!(handle.0.starts_with("fake:"));

    let req = GenerationRequest::new(models[0].id.clone(), "hello".into())
        .with_n_candidates(3);
    let mut stream = backend.generate(&handle, req).await.unwrap();

    let mut done_count = 0;
    while let Some(ev) = stream.next().await {
        if matches!(ev, GenerationEvent::Done { .. }) {
            done_count += 1;
        }
    }
    assert_eq!(done_count, 3);
}
```

- [ ] **Step 2: add tokio-stream to Cargo.toml**

Under `[dependencies]`:

```toml
tokio-stream = "0.1"
```

Under `[dev-dependencies]`:

```toml
tokio = { version = "1", features = ["rt-multi-thread", "macros", "time", "fs", "process", "test-util"] }
```

- [ ] **Step 3: run the test to verify it fails**

Run: `cargo test --test phase0_inference_backend 2>&1 | tail -15`
Expected: `error[E0432]: unresolved import ...InferenceBackend`

- [ ] **Step 4: implement the trait in `src/backends/inference/mod.rs`**

Replace the contents of `src/backends/inference/mod.rs`:

```rust
//! Inference backends (Ollama, mistral.rs, custom MLX, ...).
pub mod capabilities;
pub mod request;
pub mod response;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_stream::Stream;

use self::capabilities::BackendCapabilities;
use self::request::GenerationRequest;
use self::response::GenerationEvent;
use crate::backends::types::{ModelHandle, ModelId};

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("backend unavailable: {0}")]
    Unavailable(String),

    #[error("model not found: {0}")]
    ModelNotFound(ModelId),

    #[error("adapter not found: {0}")]
    AdapterNotFound(String),

    #[error("capability not supported: {0}")]
    CapabilityNotSupported(&'static str),

    #[error("network error: {0}")]
    Network(String),

    #[error("backend returned error: {0}")]
    Backend(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListing {
    pub id: ModelId,
    pub is_downloaded: bool,
    pub size_bytes: Option<u64>,
}

/// A backend that can generate text from a prompt.
///
/// Backends MUST:
/// - report honest capabilities (never claim a feature they do not run)
/// - stream events in candidate-index order within a single candidate,
///   but events from different candidates may interleave
/// - return `BackendError::CapabilityNotSupported` rather than silently
///   falling back to a worse path
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> BackendCapabilities;

    async fn list_models(&self) -> Result<Vec<ModelListing>, BackendError>;
    async fn load_model(&self, id: &ModelId) -> Result<ModelHandle, BackendError>;

    async fn generate(
        &self,
        handle: &ModelHandle,
        request: GenerationRequest,
    ) -> Result<Box<dyn Stream<Item = GenerationEvent> + Send + Unpin>, BackendError>;
}
```

- [ ] **Step 5: run the test**

Run: `cargo test --test phase0_inference_backend 2>&1 | tail -10`
Expected: `test result: ok. 1 passed`

- [ ] **Step 6: commit**

```bash
git add src/backends/inference/mod.rs tests/phase0_inference_backend.rs Cargo.toml Cargo.lock
git commit -m "feat(backends): InferenceBackend trait with streaming + capability negotiation"
```

### Task 0.7: `TrainingBackend` trait, `LoraConfig`, `DpoConfig`, `AdapterArtifact`

**Files:**
- Modify: `src/backends/training/mod.rs`
- Modify: `src/backends/training/config.rs`
- Modify: `src/backends/training/artefact.rs`
- Create: `tests/phase0_training_backend.rs`

- [ ] **Step 1: write the failing test**

Create `tests/phase0_training_backend.rs`:

```rust
use async_trait::async_trait;
use std::path::PathBuf;
use writer_cli::backends::training::artefact::AdapterArtifact;
use writer_cli::backends::training::config::{DpoConfig, LoraConfig, TrainingProgress};
use writer_cli::backends::training::{TrainingBackend, TrainingError};
use writer_cli::backends::types::{AdapterRef, ModelId};

struct FakeTrainer;

#[async_trait]
impl TrainingBackend for FakeTrainer {
    fn name(&self) -> &str {
        "fake-trainer"
    }

    async fn train_lora(
        &self,
        config: LoraConfig,
        _on_progress: &dyn Fn(TrainingProgress),
    ) -> Result<AdapterArtifact, TrainingError> {
        Ok(AdapterArtifact {
            adapter: AdapterRef::new(config.profile.clone(), PathBuf::from("/tmp/fake.safetensors")),
            base_model: config.base_model.clone(),
            steps: 100,
            final_loss: 1.23,
            training_seconds: 42,
        })
    }

    async fn train_dpo(
        &self,
        _config: DpoConfig,
        _on_progress: &dyn Fn(TrainingProgress),
    ) -> Result<AdapterArtifact, TrainingError> {
        Err(TrainingError::NotImplemented)
    }
}

#[tokio::test]
async fn fake_trainer_produces_artefact() {
    let trainer = FakeTrainer;
    let cfg = LoraConfig {
        profile: "default".into(),
        base_model: "google/gemma-4-26b-a4b".parse::<ModelId>().unwrap(),
        dataset_dir: PathBuf::from("/tmp/samples"),
        adapter_out: PathBuf::from("/tmp/adapter.safetensors"),
        rank: 16,
        alpha: 32.0,
        learning_rate: 1e-4,
        batch_size: 4,
        max_steps: 1000,
        max_seq_len: 4096,
    };

    let noop = |_p: TrainingProgress| {};
    let artefact = trainer.train_lora(cfg, &noop).await.unwrap();
    assert_eq!(artefact.steps, 100);
    assert!((artefact.final_loss - 1.23).abs() < 1e-6);
}
```

- [ ] **Step 2: run the test to verify it fails**

Run: `cargo test --test phase0_training_backend 2>&1 | tail -15`
Expected: `error[E0432]: unresolved import ...TrainingBackend`

- [ ] **Step 3: implement `config.rs`**

Replace `src/backends/training/config.rs`:

```rust
//! Training configuration vocabulary.
//!
//! All knobs are typed. Backends must accept or reject each field; silent
//! fallbacks for unsupported settings are a bug.
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::backends::types::ModelId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    pub profile: String,
    pub base_model: ModelId,
    pub dataset_dir: PathBuf,
    pub adapter_out: PathBuf,
    pub rank: u16,
    pub alpha: f32,
    pub learning_rate: f32,
    pub batch_size: u16,
    pub max_steps: u32,
    pub max_seq_len: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpoConfig {
    pub profile: String,
    pub base_model: ModelId,
    pub preference_dataset: PathBuf,
    pub adapter_out: PathBuf,
    pub base_adapter: Option<PathBuf>,
    pub beta: f32,
    pub learning_rate: f32,
    pub batch_size: u16,
    pub max_steps: u32,
    pub max_seq_len: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingProgress {
    pub step: u32,
    pub total_steps: u32,
    pub loss: f32,
    pub learning_rate: f32,
    pub tokens_per_second: f32,
}
```

- [ ] **Step 4: implement `artefact.rs`**

Replace `src/backends/training/artefact.rs`:

```rust
//! Training output artefact.
use serde::{Deserialize, Serialize};

use crate::backends::types::{AdapterRef, ModelId};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterArtifact {
    pub adapter: AdapterRef,
    pub base_model: ModelId,
    pub steps: u32,
    pub final_loss: f32,
    pub training_seconds: u64,
}
```

- [ ] **Step 5: implement the trait in `mod.rs`**

Replace `src/backends/training/mod.rs`:

```rust
//! Training backends (mlx-tune, unsloth, ...).
pub mod artefact;
pub mod config;

use async_trait::async_trait;

use self::artefact::AdapterArtifact;
use self::config::{DpoConfig, LoraConfig, TrainingProgress};

#[derive(Debug, thiserror::Error)]
pub enum TrainingError {
    #[error("training backend unavailable: {0}")]
    Unavailable(String),

    #[error("dataset error: {0}")]
    Dataset(String),

    #[error("training failed: {0}")]
    TrainingFailed(String),

    #[error("not implemented yet")]
    NotImplemented,

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[async_trait]
pub trait TrainingBackend: Send + Sync {
    fn name(&self) -> &str;

    async fn train_lora(
        &self,
        config: LoraConfig,
        on_progress: &dyn Fn(TrainingProgress),
    ) -> Result<AdapterArtifact, TrainingError>;

    async fn train_dpo(
        &self,
        config: DpoConfig,
        on_progress: &dyn Fn(TrainingProgress),
    ) -> Result<AdapterArtifact, TrainingError>;
}
```

- [ ] **Step 6: run the test**

Run: `cargo test --test phase0_training_backend 2>&1 | tail -10`
Expected: `test result: ok. 1 passed`

- [ ] **Step 7: commit**

```bash
git add src/backends/training/ tests/phase0_training_backend.rs
git commit -m "feat(backends): TrainingBackend trait + LoraConfig/DpoConfig/AdapterArtifact"
```

### Task 0.8: `Sample` struct in `corpus::sample`

**Files:**
- Create: `src/corpus/mod.rs`
- Create: `src/corpus/sample.rs`
- Modify: `src/lib.rs`
- Create: `tests/phase0_sample.rs`

- [ ] **Step 1: write the failing test**

Create `tests/phase0_sample.rs`:

```rust
use writer_cli::corpus::sample::{Sample, SampleSource, SampleMetadata};

#[test]
fn sample_computes_stable_content_hash() {
    let a = Sample::new(
        "hello world".into(),
        SampleMetadata {
            source: SampleSource::Markdown,
            origin_path: Some("/tmp/note.md".into()),
            context_tag: Some("longform".into()),
            captured_at: None,
        },
    );
    let b = Sample::new("hello world".into(), a.metadata.clone());
    assert_eq!(a.content_hash, b.content_hash);
}

#[test]
fn sample_hash_differs_for_different_content() {
    let a = Sample::new("hello world".into(), SampleMetadata::default());
    let b = Sample::new("hello wrld".into(), SampleMetadata::default());
    assert_ne!(a.content_hash, b.content_hash);
}

#[test]
fn sample_tokens_word_count_is_unicode_aware() {
    let s = Sample::new("les élèves étudient".into(), SampleMetadata::default());
    assert_eq!(s.word_count(), 3);
}
```

- [ ] **Step 2: run the test to verify it fails**

Run: `cargo test --test phase0_sample 2>&1 | tail -15`
Expected: `error[E0432]: unresolved import ...corpus::sample::Sample`

- [ ] **Step 3: create `src/corpus/mod.rs`**

```rust
//! Corpus pipeline: sources -> normalisation -> samples.
pub mod sample;
```

- [ ] **Step 4: create `src/corpus/sample.rs`**

```rust
//! On-disk sample representation.
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SampleSource {
    #[default]
    Unknown,
    Markdown,
    PlainText,
    TwitterArchive,
    LinkedIn,
    Obsidian,
    EmailMbox,
    Url,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SampleMetadata {
    pub source: SampleSource,
    pub origin_path: Option<PathBuf>,
    pub context_tag: Option<String>,
    pub captured_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    pub content: String,
    pub metadata: SampleMetadata,
    pub content_hash: String,
}

impl Sample {
    pub fn new(content: String, metadata: SampleMetadata) -> Self {
        let content_hash = hash_content(&content);
        Self { content, metadata, content_hash }
    }

    /// Unicode-segmenter word count, used for corpus size reports.
    pub fn word_count(&self) -> usize {
        self.content.unicode_words().count()
    }

    /// Unicode-segmenter sentence approximation, used for stylometry.
    pub fn char_count(&self) -> usize {
        self.content.chars().count()
    }
}

fn hash_content(s: &str) -> String {
    // Stable, small, deterministic hash. Not cryptographic — used for
    // dedupe only.
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}
```

- [ ] **Step 5: add `pub mod corpus;` to `src/lib.rs`**

Modify `src/lib.rs`:

```rust
//! writer-cli library surface — used by integration tests.

pub mod backends;
pub mod corpus;
pub mod stylometry;
```

(`stylometry` gets its placeholder in Task 0.9.)

- [ ] **Step 6: create a stylometry placeholder so the lib compiles**

Create `src/stylometry/mod.rs`:

```rust
//! Stylometric fingerprint computation — real work in Phase 2.
pub mod fingerprint;
```

Create `src/stylometry/fingerprint.rs`:

```rust
//! Placeholder — real computation lands in Phase 2.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StylometricFingerprint {
    pub word_count: u64,
}
```

- [ ] **Step 7: run the test**

Run: `cargo test --test phase0_sample 2>&1 | tail -10`
Expected: `test result: ok. 3 passed`

- [ ] **Step 8: commit**

```bash
git add src/corpus/ src/stylometry/ src/lib.rs tests/phase0_sample.rs
git commit -m "feat(corpus): Sample struct with content hash + unicode word count"
```

### Task 0.9: configuration schema for backend selection

**Files:**
- Modify: `src/config.rs`
- Modify: `src/commands/agent_info.rs`

- [ ] **Step 1: extend `AppConfig` with backend selection**

Modify `src/config.rs` — replace the `AppConfig` struct and its companions:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub active_profile: String,
    pub base_model: String,
    pub update: UpdateConfig,
    pub inference: InferenceConfig,
    pub decoding: DecodingConfig,
    pub training: TrainingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub backend: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub ollama_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodingConfig {
    pub n_candidates: u16,
    pub contrastive_enabled: bool,
    pub contrastive_alpha: f32,
    pub banned_word_bias: f32,
    pub preferred_word_bias: f32,
    pub kv_quant: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub backend: String,
    pub rank: u16,
    pub alpha: f32,
    pub learning_rate: f32,
    pub batch_size: u16,
    pub max_steps: u32,
    pub max_seq_len: u32,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            active_profile: "default".into(),
            base_model: "google/gemma-4-26b-a4b".into(),
            update: UpdateConfig::default(),
            inference: InferenceConfig::default(),
            decoding: DecodingConfig::default(),
            training: TrainingConfig::default(),
        }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            backend: "ollama".into(),
            temperature: 0.7,
            max_tokens: 2048,
            ollama_url: "http://localhost:11434".into(),
        }
    }
}

impl Default for DecodingConfig {
    fn default() -> Self {
        Self {
            n_candidates: 8,
            contrastive_enabled: true,
            contrastive_alpha: 0.3,
            banned_word_bias: -4.0,
            preferred_word_bias: 1.5,
            kv_quant: "auto".into(),
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            backend: "mlx-tune".into(),
            rank: 16,
            alpha: 32.0,
            learning_rate: 1e-4,
            batch_size: 4,
            max_steps: 1000,
            max_seq_len: 4096,
        }
    }
}
```

(Also remove the now-orphaned `InferenceConfig` impl Default from v0.1 if it duplicates.)

- [ ] **Step 2: update the init command to serialise the new config**

Modify `src/commands/init.rs` — replace the `toml_serialize` helper:

```rust
fn toml_serialize(config: &AppConfig) -> Result<String, AppError> {
    Ok(format!(
        "active_profile = \"{}\"\nbase_model = \"{}\"\n\n\
         [update]\nenabled = {}\nowner = \"{}\"\nrepo = \"{}\"\n\n\
         [inference]\nbackend = \"{}\"\ntemperature = {}\nmax_tokens = {}\nollama_url = \"{}\"\n\n\
         [decoding]\nn_candidates = {}\ncontrastive_enabled = {}\ncontrastive_alpha = {}\nbanned_word_bias = {}\npreferred_word_bias = {}\nkv_quant = \"{}\"\n\n\
         [training]\nbackend = \"{}\"\nrank = {}\nalpha = {}\nlearning_rate = {}\nbatch_size = {}\nmax_steps = {}\nmax_seq_len = {}\n",
        config.active_profile,
        config.base_model,
        config.update.enabled, config.update.owner, config.update.repo,
        config.inference.backend, config.inference.temperature, config.inference.max_tokens, config.inference.ollama_url,
        config.decoding.n_candidates, config.decoding.contrastive_enabled, config.decoding.contrastive_alpha,
        config.decoding.banned_word_bias, config.decoding.preferred_word_bias, config.decoding.kv_quant,
        config.training.backend, config.training.rank, config.training.alpha,
        config.training.learning_rate, config.training.batch_size, config.training.max_steps, config.training.max_seq_len,
    ))
}
```

- [ ] **Step 3: update `agent-info` manifest**

Modify `src/commands/agent_info.rs` — find the `"config"` block and replace with:

```rust
"config": {
    "path": config_path.display().to_string(),
    "profiles_dir": profiles_dir.display().to_string(),
    "models_dir": models_dir.display().to_string(),
    "env_prefix": "WRITER_",
    "inference_backend": "ollama",
    "training_backend": "mlx-tune",
    "decoding_stack": [
        "logit_bias_from_fingerprint",
        "contrastive_decoding_optional",
        "generate_n_rank",
        "post_hoc_filter"
    ]
},
```

- [ ] **Step 4: build and run agent-info to verify**

Run:
```bash
cargo build 2>&1 | tail -5
./target/debug/writer agent-info | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['config']['decoding_stack'])"
```
Expected: `['logit_bias_from_fingerprint', 'contrastive_decoding_optional', 'generate_n_rank', 'post_hoc_filter']`

- [ ] **Step 5: commit**

```bash
git add src/config.rs src/commands/init.rs src/commands/agent_info.rs
git commit -m "feat(config): add inference/decoding/training config sections with quality defaults"
```

### Task 0.10: Phase 0 gate — all tests pass, binary works

- [ ] **Step 1: run the full test suite**

Run: `cargo test 2>&1 | tail -20`
Expected: all tests pass, at minimum:
  - `tests/phase0_types.rs` — 9 tests
  - `tests/phase0_inference_backend.rs` — 1 test
  - `tests/phase0_training_backend.rs` — 1 test
  - `tests/phase0_sample.rs` — 3 tests

- [ ] **Step 2: verify the release binary still builds and runs**

Run:
```bash
cargo build --release 2>&1 | tail -5
./target/release/writer --version
./target/release/writer agent-info | python3 -c "import sys, json; json.load(sys.stdin); print('agent-info OK')"
./target/release/writer init 2>&1 | head -5
```
Expected: build finishes, version prints, agent-info is valid JSON, init succeeds.

- [ ] **Step 3: tag the phase**

```bash
git tag phase0-complete
git log --oneline phase0-complete~14..phase0-complete
```

- [ ] **Step 4: push**

```bash
git push origin main phase0-complete
```

**Phase 0 done.** Writer now has the full type and trait vocabulary for backends, samples, and fingerprints. Every command still behaves exactly as v0.1 on the outside. The foundation is in place for Phase 1 to plug in real source parsers and Phase 3 to plug in the Ollama backend.

---

## Phase 1 — Data pipeline (task specifications)

**Goal:** Turn `writer learn` into a real ingestion pipeline with multiple source formats, normalisation, deduplication, and context tagging. Quality starts here — a LoRA fine-tune is only as good as the samples that go in.

**Files created:**
- `src/corpus/sources/mod.rs`
- `src/corpus/sources/markdown.rs`
- `src/corpus/sources/plain_text.rs`
- `src/corpus/sources/twitter_archive.rs`
- `src/corpus/sources/obsidian.rs`
- `src/corpus/sources/url.rs`
- `src/corpus/normalize.rs`
- `src/corpus/dedupe.rs`
- `src/corpus/chunker.rs`
- `tests/phase1_sources.rs`
- `tests/phase1_pipeline.rs`
- `tests/fixtures/twitter_archive_tiny.zip`
- `tests/fixtures/obsidian_vault/` (small fixture vault)

**Files modified:**
- `src/commands/learn.rs` — delegate to `corpus::pipeline::ingest`
- `src/cli.rs` — add `--context` flag to `learn` subcommand, add `--clean/--raw` flag

### Task 1.1: `Source` trait + registry

Interface contract:

```rust
#[async_trait]
pub trait Source: Send + Sync {
    fn name(&self) -> &'static str;
    fn matches(&self, path: &Path) -> bool;
    async fn parse(&self, path: &Path, context: Option<&str>) -> Result<Vec<Sample>, SourceError>;
}

pub struct SourceRegistry {
    sources: Vec<Box<dyn Source>>,
}

impl SourceRegistry {
    pub fn default_set() -> Self { /* all shipped sources */ }
    pub fn detect(&self, path: &Path) -> Option<&dyn Source>;
}
```

Test: registry with all shipped sources picks the right parser for `.md`, `.txt`, `twitter.zip`, `obsidian-vault/`.

### Task 1.2: Markdown source

- Strip YAML front matter (`^---\n.*?\n---\n`)
- Strip HTML comments
- Preserve code fences as-is but flag them with metadata (these are often not the author's voice and should be downweighted in fingerprinting)
- Split by top-level `#` headers into separate samples, tagged with the header text as a sub-context
- Accept files via glob (`*.md`) or directory walk

Tests:
- Front-matter strip preserves body
- `# Section` splitting produces multiple samples with correct tags
- Code fences are preserved but flagged

### Task 1.3: Plain text source

- Accept `.txt`
- Split by double newline into paragraphs, re-join into samples of `max_seq_len / 2` words
- Strip BOM, normalise line endings
- Detect and flag language (English vs other) via a character frequency heuristic; non-English samples get a `language` metadata tag

Tests:
- BOM stripped
- Paragraph joining respects chunk size
- Language heuristic tags `bonjour tout le monde` as not-English

### Task 1.4: Twitter archive source

- Accept `.zip` files matching the Twitter/X export format
- Extract `data/tweets.js`, parse the JSON (stripping the `window.YTD.tweets.part0 = ` prefix)
- For each tweet: skip retweets (unless `--include-retweets`), skip replies to others (unless `--include-replies`), strip `@mention` and `t.co/...` tracking links, skip tweets under 5 words
- Emit one `Sample` per tweet with `context_tag: "twitter"`

Tests:
- Fixture archive with 10 tweets, 3 retweets, 2 replies → 5 samples after filtering
- URL/mention stripping preserves substantive content
- Short tweets filtered out

### Task 1.5: Obsidian vault source

- Accept a directory containing `.obsidian/` metadata
- Walk `*.md` files recursively, skipping `.trash/` and attachments
- For each note: strip wikilinks (`[[foo]]` → `foo`), strip Dataview blocks, preserve headings
- Detect `tags: [daily]` front matter → `context_tag: "journal"`; default → `context_tag: "notes"`

Tests:
- Fixture vault with 3 notes including a daily → 3 samples with correct context tags
- Wikilinks unwrapped to plain text

### Task 1.6: URL source

- Accept HTTP/HTTPS URLs
- Fetch via reqwest (reusing the HTTP client)
- Use `scraper` to extract `<article>` or `<main>` content
- Fall back to `<body>` minus nav/footer/script/style
- Convert to plain text, preserve paragraph breaks
- Set `context_tag` from command-line argument or heuristic (e.g., `substack.com` → `"blog"`)

Tests:
- Mocked HTTP server via wiremock returns a known article, source extracts it correctly
- `<script>` and `<style>` stripped
- 404 returns a clear error

### Task 1.7: Normalisation module

`src/corpus/normalize.rs` — a chain of transformations every sample passes through:

```rust
pub fn clean(sample: Sample) -> Sample {
    sample
        .map_content(strip_signatures)
        .map_content(strip_zero_width)
        .map_content(normalize_whitespace)
        .map_content(normalize_quotes)
        .map_content(strip_tracking_params)
}
```

Each transformation is independently testable.

Tests:
- Signature block (`-- \nboris`) stripped
- Smart quotes preserved (author's choice), but mixed quotes unified per sample
- Zero-width characters removed
- Tracking params (`?utm_source=...`) stripped from URLs in text

### Task 1.8: Deduplication

`src/corpus/dedupe.rs`:
- Load existing samples from profile
- Hash incoming samples by `content_hash`
- Reject duplicates with a report
- Optional: near-duplicate detection via MinHash (default off, `--near-dedupe` opt-in)

Tests:
- Exact duplicate rejected
- Near-duplicate (same content + one extra word) allowed by default, rejected with `--near-dedupe`

### Task 1.9: Chunker

`src/corpus/chunker.rs`:
- Split long samples into chunks of `max_seq_len` tokens (approximated via unicode word count × 1.3)
- Overlap windows by 20% for context continuity
- Preserve paragraph boundaries — never split mid-paragraph

Tests:
- 10k-word sample produces multiple chunks with correct overlap
- Paragraphs are respected at chunk boundaries

### Task 1.10: `corpus::ingest` top-level pipeline

```rust
pub async fn ingest(
    paths: &[PathBuf],
    context: Option<&str>,
    config: &CorpusConfig,
    profile: &Profile,
) -> Result<IngestReport, CorpusError>;

pub struct IngestReport {
    pub samples_added: usize,
    pub samples_skipped_dedupe: usize,
    pub samples_skipped_quality: usize,
    pub total_words: usize,
    pub contexts: HashMap<String, usize>,
}
```

Integration test: ingest a fixture directory containing markdown + twitter archive + obsidian vault, verify the report shape.

### Task 1.11: Rewire `writer learn`

- Add `--context <name>` flag
- Add `--clean` (default on) and `--raw` (disable normalisation)
- Delegate entirely to `corpus::ingest`
- Print the report in human + JSON modes

Integration test via `assert_cmd`: `writer learn ./tests/fixtures/` produces the expected report.

**Phase 1 gate:** `cargo test` passes, `./target/release/writer learn ~/Documents/some-markdown/` ingests a real directory and the fingerprint updates.

---

## Phase 2 — Stylometric fingerprint v2 (task specifications)

**Goal:** Replace the current 5-metric fingerprint with a full stylometric profile that drives the decoding layer (logit bias) and the `writer score` command.

**Files created:**
- `src/stylometry/features/mod.rs`
- `src/stylometry/features/lengths.rs`
- `src/stylometry/features/function_words.rs`
- `src/stylometry/features/ngrams.rs`
- `src/stylometry/features/punctuation.rs`
- `src/stylometry/features/vocabulary.rs`
- `src/stylometry/scoring.rs`
- `src/stylometry/dashboard.rs`
- `src/stylometry/ai_slop.rs` (embedded `const` banned list)
- `tests/phase2_stylometry.rs`

**Files modified:**
- `src/stylometry/fingerprint.rs` — replace placeholder with real `StylometricFingerprint`
- `src/commands/profile.rs` — use the new fingerprint + add `--html` flag to `profile show`

### Task 2.1: length features

`lengths.rs` computes:
- Word length distribution (mean, SD, histogram via hdrhistogram)
- Sentence length distribution (mean, SD, histogram)
- Paragraph length distribution
- Uses unicode-segmentation for word and sentence boundaries

Tests: known inputs produce expected distributions within tolerance.

### Task 2.2: function word features

`function_words.rs`:
- Embedded list of the 200 most common English function words (the, a, of, and, ...)
- Compute relative frequency in the corpus
- Return a `HashMap<String, f64>` mapping word → frequency
- This is a strong stylometric signal — authors have characteristic function-word fingerprints

Tests: known corpus produces expected frequencies within tolerance.

### Task 2.3: character n-gram profile

`ngrams.rs`:
- Compute 3-gram and 4-gram frequency distribution
- Return top-500 n-grams by frequency
- Captures spelling tics, preferred apostrophes, typos

Tests: "the quick brown fox" produces expected top trigrams.

### Task 2.4: punctuation features

`punctuation.rs`:
- Count em-dashes, en-dashes, hyphens, semicolons, colons, parentheses, exclamation marks, question marks
- Normalise per 1000 words
- The em-dash count is an especially strong AI tell; the user's own em-dash rate is the target

Tests: expected counts on fixture text.

### Task 2.5: vocabulary features

`vocabulary.rs`:
- Compute vocabulary set
- Compute banned-word list: AI-slop words (from `ai_slop.rs`) NOT in user vocabulary
- Compute preferred-word list: words in user vocabulary with unusually high frequency relative to a reference corpus (use SUBTLEX frequency list as reference, embedded as a `const`)
- Return lists usable as logit bias inputs

Tests: user writing with zero instances of "delve" → banned list contains "delve"; user writing that contains "shoggoth" much more than SUBTLEX predicts → preferred list contains "shoggoth".

### Task 2.6: AI-slop word list

`ai_slop.rs`:
- Embed the full banned word list (sourced from `~/.claude/skills/humanise-text/reference/banned_patterns.md` and the llmstrip project's 280-word list)
- Expose as `const BANNED_WORDS: &[&str] = &[...]`
- Expose as `const BANNED_PHRASES: &[&str] = &[...]`

Tests: list is non-empty, no duplicates, all lowercase.

### Task 2.7: `StylometricFingerprint` struct + `compute`

Replace `src/stylometry/fingerprint.rs` placeholder with:

```rust
pub struct StylometricFingerprint {
    pub word_count: u64,
    pub char_count: u64,
    pub word_length: LengthStats,
    pub sentence_length: LengthStats,
    pub paragraph_length: LengthStats,
    pub function_words: HashMap<String, f64>,
    pub ngram_profile: Vec<(String, u64)>,
    pub punctuation: PunctuationStats,
    pub vocabulary_size: u64,
    pub banned_words: Vec<String>,
    pub preferred_words: Vec<(String, f64)>,
    pub contexts: HashMap<String, ContextStats>,
}

impl StylometricFingerprint {
    pub fn compute(samples: &[Sample]) -> Self;
    pub fn logit_bias_map(&self, config: &DecodingConfig) -> LogitBiasMap;
}
```

Tests: round-trip serialisation, compute produces reasonable output on fixture corpus.

### Task 2.8: `scoring::distance(text, fingerprint)`

`scoring.rs`:
- Computes a 0-1 stylometric distance between a text sample and a fingerprint
- Weighted combination of: sentence-length KL divergence, function-word cosine distance, punctuation L1 distance, n-gram cosine distance, AI-slop word penalty
- Returns a `DistanceReport` with per-component scores

Tests: a sample from the fingerprint's own corpus scores below 0.2; an AI-generated sample scores above 0.6.

### Task 2.9: HTML dashboard

`dashboard.rs`:
- Produce a single HTML file with embedded CSS and SVG charts
- No JS, no external assets — fully offline
- Sections: overview, length distributions, function word deviation from reference, n-gram highlights, banned/preferred lists
- Writable to any path via `writer profile show --html path.html`

Tests: HTML output contains expected sections; file is valid HTML5.

### Task 2.10: `writer profile show --html` command

- Add `--html <path>` flag to `profile show`
- Call `dashboard::render(&fingerprint, path)`
- Print the path in the success response

Tests: assert_cmd run produces the file.

**Phase 2 gate:** `writer profile show` prints all new metrics, `writer profile show --html /tmp/fp.html` produces a valid dashboard, and `writer score` (to be wired in Phase 8) can be prototyped against the scoring module.

---

## Phase 3 — Ollama inference backend (task specifications)

**Goal:** Implement `OllamaBackend: InferenceBackend` so `writer write` and `writer rewrite` can produce real text. After this phase, the user experience is: `writer init && writer learn . && writer write "..."` and text comes out.

**Files created:**
- `src/backends/inference/ollama.rs`
- `tests/phase3_ollama.rs` (via wiremock)

**Files modified:**
- `src/backends/inference/mod.rs` — add `pub mod ollama;`
- `src/commands/write.rs` — replace stub with real implementation
- `src/commands/rewrite.rs` — replace stub with real implementation
- `src/commands/model.rs` — wire `model list` and `model pull` to `OllamaBackend`

### Task 3.1: Ollama HTTP client basics

- Build a `reqwest::Client` with the configured `ollama_url`
- Implement `ping()` returning Ollama version, used by `writer init` for diagnostics
- Map network failures to `BackendError::Unavailable` with actionable suggestion text

### Task 3.2: `list_models` via `GET /api/tags`

- Parse the Ollama tag list response
- Map Ollama model names back to `ModelId` (`google/gemma-4-26b-a4b` ↔ `gemma3:26b-a4b` or similar — maintain a mapping table)
- Test via wiremock

### Task 3.3: `load_model` with pull-on-demand

- Call `POST /api/pull` if the model is not listed
- Stream progress via `GenerationProgress` events (reuse the training progress type? No — define an inference-side `PullProgress`)
- Return a `ModelHandle` once pull completes

### Task 3.4: `generate` non-streaming (baseline)

- Call `POST /api/generate` with the prompt, model, options
- Translate `GenerationRequest` → Ollama options
- Map `num_predict`, `temperature`, `top_p`, `top_k`, `repeat_penalty`
- Collect full response, emit `Token` + `Done` events from a single completed response

Tests via wiremock: request body matches spec, response parses correctly.

### Task 3.5: `generate` streaming via chunked NDJSON

- Switch to `stream=true` path
- Parse NDJSON response line by line
- Emit `Token` events as tokens arrive
- Emit `Done` on final chunk

### Task 3.6: Multi-candidate generation

- Ollama does not natively support N candidates in one call — run N concurrent requests
- Emit events with `candidate_index` tagged per request
- Use a `tokio::task::JoinSet` to parallelise

### Task 3.7: LoRA adapter loading via Modelfile

- When `request.adapter` is set, generate a Modelfile on the fly:
  ```
  FROM gemma3:26b-a4b
  ADAPTER /path/to/adapter.safetensors
  ```
- Register the custom model via `POST /api/create` with name `writer-<profile>`
- Use the custom model name in the generate call
- Cache: if the adapter file's mtime matches the registered model's created_at, reuse; otherwise re-register

### Task 3.8: Logit bias via Ollama options

- Ollama supports `logit_bias` as a map of token IDs to biases (version ≥ 0.5.2)
- Tokenise bias keys using a local tokenizer matching the model (`tokenizers` crate, cached tokenizer files per model)
- Translate the fingerprint's `banned_words` and `preferred_words` into token ID bias map
- If the installed Ollama version does not support logit bias, return `CapabilityNotSupported`

### Task 3.9: Capability reporting

- Implement `capabilities()` by probing the connected Ollama version
- `GET /api/version` returns semver; the module has a table of min-version-per-feature
- `supports_lora` = always true
- `supports_logit_bias` = Ollama ≥ 0.5.2
- `kv_quant = KvQuantKind::TurboQuant` = Ollama ≥ (TBD when it lands)
- `quant_schemes` = probe the tag list for known patterns (Q4_K_M, Q8_0, etc.)

### Task 3.10: Wire `writer write` to `OllamaBackend`

- `commands::write::run(ctx, prompt)` — load config, construct `GenerationRequest`, call `decoding::run` (stub that just calls backend for now), print output
- No decoding layer yet — that's Phase 4
- Print the best candidate to stdout

### Task 3.11: Wire `writer rewrite` to `OllamaBackend`

- Same as write but with a rewrite prompt template (`templates::rewrite` — embedded const)
- Support `--in-place` to write back to the file

### Task 3.12: Wire `writer model list/pull`

- `model list` → `backend.list_models()` → table or JSON
- `model pull` → `backend.load_model()` with progress streaming to stderr

**Phase 3 gate:** With Ollama installed and running, `writer init && writer write "hello"` produces real text. The entire pipeline works end-to-end, minus the quality layer (which arrives in Phase 4).

---

## Phase 4 — Quality decoding layer (module + contract)

**Goal:** Wrap every `InferenceBackend::generate` call in a quality pipeline. No `writer write` ever skips this layer.

**Module contract:**

```rust
pub mod decoding;

pub async fn run(
    backend: &dyn InferenceBackend,
    fingerprint: &StylometricFingerprint,
    config: &DecodingConfig,
    prompt_kind: PromptKind,  // Write, Rewrite, Continue
    user_input: &str,
) -> Result<GenerationResult, DecodingError>;
```

**Pipeline:**

1. **Build the prompt** via `prompts::render(kind, user_input, fingerprint)`. Inject a small stylometric priming block (avg sentence length, top 5 preferred words, contextual tag).
2. **Construct the logit bias** via `logit_bias::from_fingerprint(fingerprint, config)`.
3. **Capability-check the backend.** If `config.contrastive_enabled` and backend doesn't support it, return a clear error with suggestion `writer config set decoding.contrastive_enabled false` or `writer config set inference.backend <other>`.
4. **Call `backend.generate`** with N candidates, logit bias, optional contrastive base.
5. **Rank candidates** via `ranker::rank(candidates, fingerprint)`. Scoring is `stylometry::scoring::distance` + length preference + AI-slop penalty.
6. **Filter the top candidate** via `filter::validate(best, fingerprint, config)`. If validation fails (sentence-length SD too low, banned words present, structural patterns detected), regenerate up to 3 times. If still failing, return the best with a warning in the metadata.
7. **Return** a `GenerationResult` containing the text, the stylometric distance, the chosen candidate index, the discarded candidates, and the timing.

**Files:** `src/decoding/mod.rs`, `logit_bias.rs`, `contrastive.rs` (calls backend twice + subtracts in the ranker), `ranker.rs`, `filter.rs`, `prompts.rs`.

**Acceptance criteria:**
- Generating 8 candidates and ranking by fingerprint distance is measurable and deterministic given the same seed
- Banned words never appear in the final output (enforced by filter as a hard check)
- A test harness produces a before/after stylometric distance comparison on a fixture corpus; post-decoding distance MUST be lower than the raw backend output by at least 30% on average

**Expansion:** Full TDD task breakdown written when Phase 4 begins.

---

## Phase 5 — mlx-tune training backend (module + contract)

**Goal:** Ship a working `writer train` on Apple Silicon that produces a LoRA adapter attached to the active profile.

**Backend contract:**
- `MlxTuneBackend` detects `mlx_lm.lora` on PATH at construction time; if missing, `new()` returns `TrainingError::Unavailable` with a suggestion pointing at `pip install mlx-tune` or `uvx mlx-tune`.
- `train_lora(config, on_progress)`:
  1. Serialise `config.dataset_dir` samples to `<tmp>/train.jsonl` in MLX chat format
  2. Write `<tmp>/lora_config.yaml` matching mlx-tune's schema
  3. Spawn `mlx_lm.lora --config <yaml>` via tokio::process
  4. Parse stdout JSON progress lines, call `on_progress`
  5. On completion, move `<tmp>/adapters/adapter.safetensors` to `config.adapter_out`
  6. Compute training metadata (steps, final_loss from log) and return `AdapterArtifact`
- `train_dpo(config, on_progress)`: spawns `mlx_lm.dpo` (or returns `NotImplemented` if the user's mlx-tune version doesn't include it)

**Acceptance criteria:**
- On a machine with mlx-tune installed, `writer train` against a 10-sample fixture completes in under 10 minutes and produces an adapter file
- Progress is streamed to the terminal in human mode, as newline-delimited JSON in `--json` mode
- Cancellation via `Ctrl-C` terminates the child process cleanly (no zombie MLX processes)

**Expansion:** Full TDD breakdown when Phase 5 begins.

---

## Phase 6 — DPO against AI-rewrites (module + contract)

**Goal:** Teach the model what _not_ to do. For each user sample, generate an AI rewrite of the same content via the base model, then train DPO with (user = preferred, AI = rejected).

**Pipeline:**

1. `writer train --with-contrast` triggers the DPO path
2. For each sample in the profile, call `InferenceBackend::generate` on the base model with a prompt like: `"Rewrite the following in a clear, professional style.\n\nORIGINAL:\n{sample}\n\nREWRITE:"` — producing the AI version
3. Build a preference dataset of `{chosen: original, rejected: ai_rewrite}` pairs
4. Call `backends::training::mlx_tune::train_dpo` on the preference dataset
5. Attach the resulting DPO adapter as a _second_ adapter layered on top of the LoRA SFT adapter (or retrained from the SFT base)

**Non-negotiable quality property:** DPO adapter must reduce stylometric distance on a held-out sample set compared to the SFT-only adapter. `writer train --with-contrast` runs the evaluation automatically and reports the delta.

**Expansion:** TDD tasks when Phase 6 begins.

---

## Phase 7 — Contrastive decoding (CoPe) (module + contract)

**Goal:** At decode time, subtract the base model's logits from the fine-tuned model's logits. Paper reference: [Personalized LLM Decoding via Contrasting Personal Preference (EMNLP 2025)](https://arxiv.org/html/2506.12109v2).

**Implementation approach:**

Ollama does not expose raw logits in v0.19. Two paths:

**Path A (preferred):** Wait for Ollama to add a `/api/generate?return_logprobs=true` endpoint. If it's already there by the time we hit Phase 7, use it.

**Path B:** Shell-out to `mlx_lm.generate` Python for contrastive decoding specifically, behind a feature flag. Writer advertises `supports_contrastive_decoding: true` only when path A is available OR the user has set `WRITER_ALLOW_PYTHON=1`.

**Decoder equation:**

```
logits_final = (1 + alpha) * logits_finetuned - alpha * logits_base
```

where `alpha` is `config.decoding.contrastive_alpha` (default 0.3).

**Acceptance criteria:**
- When enabled, stylometric distance reduction on held-out samples is measurable (target: additional 15%+ reduction beyond SFT+DPO)
- When the backend cannot support it, writer prints a clear error rather than silently running without contrastive decoding

**Expansion:** TDD tasks when Phase 7 begins.

---

## Phase 8 — `writer score` stylometric distance (module + contract)

**Goal:** A CI-friendly command that reports the stylometric distance between a text file and the active profile's fingerprint.

**Command:**

```bash
writer score draft.md
writer score draft.md --threshold 0.3 --fail
writer score draft.md --json
```

**Output (human):**
```
Profile:        default
File:           draft.md
Words:          1,234
Overall distance: 0.18 (CLOSE)

Breakdown:
  sentence length KL:   0.12
  function words cos:   0.08
  punctuation L1:       0.15
  n-gram cos:           0.22
  AI-slop penalty:      0.00

Banned words found: none
Preferred words found: 12/40
```

**Output (JSON):**
```json
{
  "version": "1",
  "status": "success",
  "data": {
    "file": "draft.md",
    "profile": "default",
    "distance": 0.18,
    "components": { ... },
    "banned_found": [],
    "preferred_found": ["shoggoth", "yonder", ...],
    "verdict": "close"
  }
}
```

**Non-negotiable:** `--fail --threshold <x>` returns exit code 1 when distance > threshold. Enables use as a git pre-commit hook or CI check.

**Expansion:** TDD tasks when Phase 8 begins.

---

## Phase 9 — Extensibility reference + docs (module + contract)

**Goal:** Prove the architecture by adding a second inference backend and documenting the process. Do this explicitly as the final phase so future work (TurboQuant, TriAttention, mistral.rs, custom MLX) is unblocked.

**Deliverables:**

1. **`MlxLmBackend`** — a second inference backend that shells out to Python `mlx_lm.generate`. Not for production use (Python dep), but for cutting-edge feature access. Must implement the same `InferenceBackend` trait with zero changes to the trait.

2. **`docs/backends.md`** — step-by-step guide to adding a new inference backend:
   - Implement the trait
   - Report honest capabilities
   - Add to the registry
   - Write integration tests
   - Benchmark against the reference

3. **`docs/kv_quant.md`** — step-by-step guide to wiring a new KV cache optimisation (TurboQuant, TriAttention) into an existing backend:
   - Extend `KvQuantKind` enum
   - Update capability detection
   - Translate `GenerationParams.kv_quant_preference` → backend-native options
   - Add a benchmark script
   - Example: what the TurboQuant-in-Ollama integration will look like once Ollama ships it

4. **Benchmark suite** `benches/phases.rs` — runs inference on a fixture corpus through each backend and reports tokens/sec + stylometric distance. Used to verify no phase regresses quality or speed.

**Acceptance criteria:**
- `MlxLmBackend` compiles, runs locally with `mlx-lm` installed, and passes the same integration tests as `OllamaBackend`
- Adding a hypothetical third backend takes under 2 hours of engineering following `docs/backends.md`
- The benchmark suite runs in CI and fails the build on any >5% regression

**Expansion:** TDD tasks when Phase 9 begins.

---

## Self-Review

**Spec coverage check:**
- ✅ Quality-first defaults — hardcoded in `GenerationParams::default` and `DecodingConfig::default`
- ✅ Extension points for TurboQuant / TriAttention — `KvQuantKind` enum + `BackendCapabilities.kv_quant` field (Task 0.3)
- ✅ Single static binary — no dynamic linking, training is a shell-out, inference is HTTP
- ✅ No silent fallbacks — `BackendError::CapabilityNotSupported` explicit path
- ✅ Data pipeline with normalisation — Phase 1
- ✅ Stylometric fingerprint with scoring — Phase 2
- ✅ Ollama backend — Phase 3
- ✅ Decoding layer with logit bias + generate-N-rank + filter — Phase 4
- ✅ LoRA training via mlx-tune — Phase 5
- ✅ DPO against AI-rewrites — Phase 6
- ✅ Contrastive decoding — Phase 7
- ✅ `writer score` — Phase 8
- ✅ Extensibility proof + docs — Phase 9

**Placeholder scan:** Phase 0-3 sections are explicit about file paths, interfaces, and test shapes. Phases 4-9 are tagged as "Module + contract" and explicitly deferred to per-phase TDD expansion at the time each phase starts; this is by design — expanding them now would produce 200+ more tasks without benefit.

**Type consistency:**
- `ModelId` used consistently in `request.rs`, `response.rs`, `config.rs`, `mlx_tune.rs`
- `BackendCapabilities` used consistently; every mention agrees on the field names in Task 0.3
- `LoraConfig` field names consistent between Task 0.7 definition and Phase 5 usage

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-09-writer-quality-architecture.md`.

Phase 0 is ready to execute immediately (14 bite-sized tasks, roughly 2-4 hours of focused work, all TDD with explicit tests). Phases 1-3 are specified at task level and can be executed in sequence after Phase 0 lands. Phases 4-9 are module contracts, ready to be expanded into full TDD plans when we reach them.

Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task in Phase 0, review between tasks, fast iteration. Use `superpowers:subagent-driven-development`.

2. **Inline Execution** — work through Phase 0 in this session, batch execution with checkpoints. Use `superpowers:executing-plans`.

Which approach?
