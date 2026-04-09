use serde::Serialize;
use std::path::PathBuf;

use crate::config;
use crate::error::AppError;
use crate::output::{self, Ctx};

#[derive(Serialize)]
struct LearnResult {
    profile: String,
    files_added: Vec<String>,
    files_skipped: Vec<SkippedFile>,
    total_words: usize,
    total_chars: usize,
    samples_dir: String,
}

#[derive(Serialize)]
struct SkippedFile {
    path: String,
    reason: String,
}

pub fn run(ctx: Ctx, files: Vec<PathBuf>) -> Result<(), AppError> {
    let cfg = config::load()?;
    let profile_dir = config::profiles_dir().join(&cfg.active_profile);
    let samples_dir = profile_dir.join("samples");

    if !samples_dir.exists() {
        return Err(AppError::Config(format!(
            "Profile '{}' does not exist. Run: writer init",
            cfg.active_profile
        )));
    }

    let mut files_added = Vec::new();
    let mut files_skipped = Vec::new();
    let mut total_words = 0;
    let mut total_chars = 0;

    for file in files {
        if !file.exists() {
            files_skipped.push(SkippedFile {
                path: file.display().to_string(),
                reason: "file_not_found".into(),
            });
            continue;
        }

        let content = match std::fs::read_to_string(&file) {
            Ok(c) => c,
            Err(e) => {
                files_skipped.push(SkippedFile {
                    path: file.display().to_string(),
                    reason: format!("not_utf8_or_unreadable: {e}"),
                });
                continue;
            }
        };

        let file_name = file
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "sample.txt".into());
        let dest = samples_dir.join(&file_name);
        std::fs::write(&dest, &content)?;

        total_words += content.split_whitespace().count();
        total_chars += content.chars().count();
        files_added.push(dest.display().to_string());
    }

    let result = LearnResult {
        profile: cfg.active_profile.clone(),
        files_added,
        files_skipped,
        total_words,
        total_chars,
        samples_dir: samples_dir.display().to_string(),
    };

    output::print_success_or(ctx, &result, |r| {
        use owo_colors::OwoColorize;
        println!(
            "{} Added {} file(s) to profile '{}'",
            "+".green(),
            r.files_added.len(),
            r.profile.bold()
        );
        println!(
            "  {} words, {} chars",
            r.total_words.to_string().bold(),
            r.total_chars.to_string().bold()
        );
        if !r.files_skipped.is_empty() {
            println!(
                "  {} {} file(s) skipped",
                "!".yellow(),
                r.files_skipped.len()
            );
            for s in &r.files_skipped {
                println!("    {} ({})", s.path.dimmed(), s.reason.dimmed());
            }
        }
        println!();
        println!("Next: {}", "writer train".bold());
    });

    Ok(())
}
