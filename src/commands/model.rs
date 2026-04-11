use serde::Serialize;

use crate::config;
use crate::error::AppError;
use crate::output::{self, Ctx};
use writer_cli::backends::inference::InferenceBackend;
use writer_cli::backends::inference::ollama::OllamaBackend;
use writer_cli::backends::types::ModelId;

#[derive(Serialize)]
struct ModelEntry {
    name: String,
    downloaded: bool,
    size_bytes: Option<u64>,
}

pub async fn list(ctx: Ctx) -> Result<(), AppError> {
    let cfg = config::load()?;
    let backend = OllamaBackend::new(&cfg.inference.ollama_url);

    backend
        .ping()
        .await
        .map_err(|e| AppError::Transient(e.to_string()))?;

    let models = backend
        .list_models()
        .await
        .map_err(|e| AppError::Transient(e.to_string()))?;

    let entries: Vec<ModelEntry> = models
        .into_iter()
        .map(|m| ModelEntry {
            name: m.id.to_string(),
            downloaded: m.is_downloaded,
            size_bytes: m.size_bytes,
        })
        .collect();

    output::print_success_or(ctx, &entries, |list| {
        use owo_colors::OwoColorize;
        let mut table = comfy_table::Table::new();
        table.set_header(vec!["Model", "Downloaded", "Size"]);
        for m in list {
            let size = m
                .size_bytes
                .map(|b| format!("{:.1} GB", b as f64 / 1_000_000_000.0))
                .unwrap_or_else(|| "unknown".into());
            table.add_row(vec![
                m.name.clone(),
                if m.downloaded {
                    "yes".green().to_string()
                } else {
                    "no".dimmed().to_string()
                },
                size,
            ]);
        }
        println!("{table}");
    });

    Ok(())
}

pub async fn pull(ctx: Ctx, name: String) -> Result<(), AppError> {
    let cfg = config::load()?;
    let backend = OllamaBackend::new(&cfg.inference.ollama_url);

    backend
        .ping()
        .await
        .map_err(|e| AppError::Transient(e.to_string()))?;

    let model_id: ModelId = name
        .parse()
        .map_err(|e| AppError::InvalidInput(format!("Invalid model name: {e}")))?;

    if !ctx.format.is_json() {
        use owo_colors::OwoColorize;
        println!("{} pulling {}...", ">".blue(), model_id);
    }

    let handle = backend
        .load_model(&model_id)
        .await
        .map_err(|e| AppError::Transient(e.to_string()))?;

    #[derive(Serialize)]
    struct PullResult {
        model: String,
        handle: String,
    }

    let result = PullResult {
        model: model_id.to_string(),
        handle: handle.0,
    };

    output::print_success_or(ctx, &result, |r| {
        use owo_colors::OwoColorize;
        println!("{} model ready: {}", "+".green(), r.model);
    });

    Ok(())
}
