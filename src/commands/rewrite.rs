use serde::Serialize;
use std::path::PathBuf;
use tokio_stream::StreamExt;

use crate::config;
use crate::error::AppError;
use crate::output::{self, Ctx};
use writer_cli::backends::inference::InferenceBackend;
use writer_cli::backends::inference::ollama::OllamaBackend;
use writer_cli::backends::inference::request::GenerationRequest;
use writer_cli::backends::inference::response::GenerationEvent;
use writer_cli::backends::types::ModelId;

#[derive(Serialize)]
struct RewriteResult {
    text: String,
    original_path: String,
    in_place: bool,
    model: String,
}

const REWRITE_PROMPT: &str = "Rewrite the following text while preserving its meaning \
and the author's voice. Keep the same level of formality, the same vocabulary preferences, \
and the same sentence rhythm. Do not add flourishes, transitions, or structure the original \
did not have. Output only the rewritten text, nothing else.\n\nORIGINAL:\n";

pub async fn run(ctx: Ctx, file: PathBuf, in_place: bool) -> Result<(), AppError> {
    let content = std::fs::read_to_string(&file)?;
    let cfg = config::load()?;
    let backend = OllamaBackend::new(&cfg.inference.ollama_url);

    backend
        .ping()
        .await
        .map_err(|e| AppError::Transient(e.to_string()))?;

    let model_id: ModelId = cfg
        .base_model
        .parse()
        .map_err(|e| AppError::Config(format!("Invalid base_model in config: {e}")))?;

    let handle = backend
        .load_model(&model_id)
        .await
        .map_err(|e| AppError::Transient(e.to_string()))?;

    let prompt = format!("{REWRITE_PROMPT}{content}\n\nREWRITTEN:");

    let req = GenerationRequest::new(model_id.clone(), prompt).with_n_candidates(1);

    let mut stream = backend
        .generate(&handle, req)
        .await
        .map_err(|e| AppError::Transient(e.to_string()))?;

    let mut rewritten = String::new();
    while let Some(event) = stream.next().await {
        match event {
            GenerationEvent::Done { full_text, .. } => {
                rewritten = full_text;
            }
            GenerationEvent::Error { message, .. } => {
                return Err(AppError::Transient(message));
            }
            _ => {}
        }
    }

    if in_place {
        std::fs::write(&file, &rewritten)?;
    }

    let result = RewriteResult {
        text: rewritten.clone(),
        original_path: file.display().to_string(),
        in_place,
        model: model_id.to_string(),
    };

    output::print_success_or(ctx, &result, |r| {
        if !in_place {
            print!("{}", r.text);
        } else {
            use owo_colors::OwoColorize;
            println!("{} rewritten in place: {}", "+".green(), r.original_path);
        }
    });

    Ok(())
}
