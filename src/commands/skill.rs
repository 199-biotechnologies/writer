use serde::Serialize;
use std::path::PathBuf;

use crate::error::AppError;
use crate::output::{self, Ctx};

// ── Skill content ───────────────────────────────────────────────────────────

fn skill_content() -> &'static str {
    r#"---
name: writer
description: >
  Local AI that writes in your voice. Feed it your own writing samples,
  fine-tune a small local model, and generate text that sounds like you.
  Run `writer agent-info` for the full capability manifest, flags, and
  exit codes.
---

## writer

`writer` is a local AI CLI that writes in your voice. It runs a small model
on your machine, fine-tuned on samples of your own writing. No cloud, no
API keys, no training data leaks.

Run `writer agent-info` for the full capability manifest.

Typical flow:

1. `writer init` -- one-time setup
2. `writer learn ~/Documents/my-writing/*.md` -- ingest samples
3. `writer train` -- fine-tune on the samples
4. `writer write "a blog post about X"` -- generate in your voice
"#
}

// ── Platform targets ────────────────────────────────────────────────────────

struct SkillTarget {
    name: &'static str,
    path: PathBuf,
}

fn home() -> PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

fn skill_targets() -> Vec<SkillTarget> {
    let h = home();
    vec![
        SkillTarget {
            name: "Claude Code",
            path: h.join(".claude/skills/writer"),
        },
        SkillTarget {
            name: "Codex CLI",
            path: h.join(".codex/skills/writer"),
        },
        SkillTarget {
            name: "Gemini CLI",
            path: h.join(".gemini/skills/writer"),
        },
    ]
}

// ── Install ─────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct InstallResult {
    platform: String,
    path: String,
    status: String,
}

pub fn install(ctx: Ctx) -> Result<(), AppError> {
    let content = skill_content();
    let mut results: Vec<InstallResult> = Vec::new();

    for target in &skill_targets() {
        let skill_path = target.path.join("SKILL.md");

        if skill_path.exists() && std::fs::read_to_string(&skill_path).is_ok_and(|c| c == content) {
            results.push(InstallResult {
                platform: target.name.into(),
                path: skill_path.display().to_string(),
                status: "already_current".into(),
            });
            continue;
        }

        std::fs::create_dir_all(&target.path)?;
        std::fs::write(&skill_path, content)?;
        results.push(InstallResult {
            platform: target.name.into(),
            path: skill_path.display().to_string(),
            status: "installed".into(),
        });
    }

    output::print_success_or(ctx, &results, |r| {
        use owo_colors::OwoColorize;
        for item in r {
            let marker = if item.status == "installed" { "+" } else { "=" };
            println!(
                " {} {} -> {}",
                marker.green(),
                item.platform.bold(),
                item.path.dimmed()
            );
        }
    });

    Ok(())
}

// ── Status ──────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct SkillStatus {
    platform: String,
    installed: bool,
    current: bool,
}

pub fn status(ctx: Ctx) -> Result<(), AppError> {
    let content = skill_content();
    let mut results: Vec<SkillStatus> = Vec::new();

    for target in &skill_targets() {
        let skill_path = target.path.join("SKILL.md");
        let (installed, current) = if skill_path.exists() {
            let current = std::fs::read_to_string(&skill_path).is_ok_and(|c| c == content);
            (true, current)
        } else {
            (false, false)
        };
        results.push(SkillStatus {
            platform: target.name.into(),
            installed,
            current,
        });
    }

    output::print_success_or(ctx, &results, |r| {
        use owo_colors::OwoColorize;
        let mut table = comfy_table::Table::new();
        table.set_header(vec!["Platform", "Installed", "Current"]);
        for item in r {
            table.add_row(vec![
                item.platform.clone(),
                if item.installed {
                    "Yes".green().to_string()
                } else {
                    "No".red().to_string()
                },
                if item.current {
                    "Yes".green().to_string()
                } else {
                    "No".dimmed().to_string()
                },
            ]);
        }
        println!("{table}");
    });

    Ok(())
}
