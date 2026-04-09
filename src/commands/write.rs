use serde::Serialize;
use tokio_stream::StreamExt;

use writer_cli::backends::inference::ollama::OllamaBackend;
use writer_cli::backends::inference::request::GenerationRequest;
use writer_cli::backends::inference::response::GenerationEvent;
use writer_cli::backends::inference::InferenceBackend;
use writer_cli::backends::types::ModelId;
use crate::config;
use crate::error::AppError;
use crate::output::{self, Ctx};

#[derive(Serialize)]
struct WriteResult {
    text: String,
    model: String,
    tokens_generated: u32,
    elapsed_ms: u64,
}

pub async fn run(ctx: Ctx, prompt: String) -> Result<(), AppError> {
    let cfg = config::load()?;
    let backend = OllamaBackend::new(&cfg.inference.ollama_url);

    // Verify Ollama is running
    backend.ping().await.map_err(|e| AppError::Transient(e.to_string()))?;

    let model_id: ModelId = cfg
        .base_model
        .parse()
        .map_err(|e| AppError::Config(format!("Invalid base_model in config: {e}")))?;

    let handle = backend
        .load_model(&model_id)
        .await
        .map_err(|e| AppError::Transient(e.to_string()))?;

    let req = GenerationRequest::new(model_id.clone(), prompt)
        .with_n_candidates(1); // Single candidate for now; Phase 4 adds rank-N

    let mut stream = backend
        .generate(&handle, req)
        .await
        .map_err(|e| AppError::Transient(e.to_string()))?;

    let mut full_text = String::new();
    let mut tokens_generated = 0u32;
    let mut elapsed_ms = 0u64;

    while let Some(event) = stream.next().await {
        match event {
            GenerationEvent::Token { text, .. } => {
                if !ctx.format.is_json() {
                    print!("{text}");
                }
                full_text = text;
            }
            GenerationEvent::Done {
                usage, full_text: ft, ..
            } => {
                full_text = ft;
                tokens_generated = usage.generated_tokens;
                elapsed_ms = usage.elapsed_ms;
            }
            GenerationEvent::Error { message, .. } => {
                return Err(AppError::Transient(message));
            }
        }
    }

    if !ctx.format.is_json() {
        println!();
    }

    let result = WriteResult {
        text: full_text,
        model: model_id.to_string(),
        tokens_generated,
        elapsed_ms,
    };

    if ctx.format.is_json() {
        output::print_success_or(ctx, &result, |_| {});
    }

    Ok(())
}
