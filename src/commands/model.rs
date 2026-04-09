use serde::Serialize;

use crate::error::AppError;
use crate::output::{self, Ctx};

#[derive(Serialize)]
struct ModelEntry {
    name: String,
    params: String,
    size: String,
    downloaded: bool,
}

pub fn list(ctx: Ctx) -> Result<(), AppError> {
    let models = vec![
        ModelEntry {
            name: "llama-3.2-1b-instruct".into(),
            params: "1B".into(),
            size: "~700 MB".into(),
            downloaded: false,
        },
        ModelEntry {
            name: "llama-3.2-3b-instruct".into(),
            params: "3B".into(),
            size: "~2 GB".into(),
            downloaded: false,
        },
        ModelEntry {
            name: "qwen2.5-1.5b-instruct".into(),
            params: "1.5B".into(),
            size: "~900 MB".into(),
            downloaded: false,
        },
        ModelEntry {
            name: "qwen2.5-3b-instruct".into(),
            params: "3B".into(),
            size: "~1.9 GB".into(),
            downloaded: false,
        },
    ];

    output::print_success_or(ctx, &models, |list| {
        use owo_colors::OwoColorize;
        let mut table = comfy_table::Table::new();
        table.set_header(vec!["Model", "Params", "Size", "Downloaded"]);
        for m in list {
            table.add_row(vec![
                m.name.clone(),
                m.params.clone(),
                m.size.clone(),
                if m.downloaded {
                    "yes".green().to_string()
                } else {
                    "no".dimmed().to_string()
                },
            ]);
        }
        println!("{table}");
    });

    Ok(())
}

pub fn pull(_ctx: Ctx, _name: String) -> Result<(), AppError> {
    Err(AppError::Transient(
        "model pull: model download is on the v0.2 milestone. See https://github.com/199-biotechnologies/writer/milestones".into()
    ))
}
