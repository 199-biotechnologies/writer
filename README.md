<div align="center">

# writer -- Local AI that writes in your voice

**A CLI that fine-tunes a small local model on your own writing, so the text it generates sounds like you wrote it. Not like ChatGPT. Not like Claude. You.**

<br />

[![Star this repo](https://img.shields.io/github/stars/199-biotechnologies/writer?style=for-the-badge&logo=github&label=%E2%AD%90%20Star%20this%20repo&color=yellow)](https://github.com/199-biotechnologies/writer/stargazers)
&nbsp;&nbsp;
[![Follow @longevityboris](https://img.shields.io/badge/Follow_%40longevityboris-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/longevityboris)

<br />

[![Rust](https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![MSRV 1.85+](https://img.shields.io/badge/MSRV-1.85%2B-orange?style=for-the-badge)](https://www.rust-lang.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
[![Local First](https://img.shields.io/badge/Local-First-brightgreen?style=for-the-badge)](#privacy)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen?style=for-the-badge)](CONTRIBUTING.md)

---

Every AI writing tool sounds like every other AI writing tool. Same em-dashes. Same "Moreover, it is important to note that." Same rule-of-three bullet points. Same sycophantic open, same tidy bow at the end. The reason is simple: they all trained on the same pile of web text, so they all fall into the same attractor. `writer` goes the other way. You hand it a folder of your own writing and it fine-tunes a small model on your voice -- your vocabulary, your rhythm, your typos, your habits of punctuation. The output sounds like you because the model learned from you.

And because the model is small (1B-3B parameters) and runs locally, your writing samples never leave your laptop. No OpenAI. No Anthropic. No "we use your data to improve our services."

[Why this exists](#why-this-exists) | [Install](#install) | [Quick start](#quick-start) | [How it works](#how-it-works) | [Commands](#commands) | [Roadmap](#roadmap) | [Privacy](#privacy)

</div>

---

## Why this exists

AI writing has a detectable fingerprint. Run any model's output through a stylometric scanner and the same tells show up: em-dashes instead of commas, transitions starting with "Moreover," paragraphs built around the rule of three, sentences with identical rhythm, closing lines that wrap the whole thing in a neat bow. The "humanise this text" industry exists because the tells are that obvious.

Humanising after the fact is a patch. You are fighting the model's entire training distribution on every call. The real fix is to train a model that never had those tells in the first place -- one that learned what sentences look like from *your* writing, not from the open web.

That is what `writer` does. It treats your voice as a distribution you already own. Feed it 5-50k words of text you wrote. It extracts a stylometric fingerprint, trains a LoRA adapter on a small base model, and from then on every `writer write "..."` call is a sample from *your* distribution. No system prompt tricks. No "rewrite this to sound like me" prompts. The model just sounds like you now.

The bonus is privacy. Your writing samples, the adapter, the inference -- all of it lives on your machine. No API key. No training data upload. `rm -rf ~/.local/share/writer` resets everything.

## Install

`writer` is a single static Rust binary. Pick one:

```bash
# From source (works today)
cargo install --git https://github.com/199-biotechnologies/writer

# From crates.io (note: package name is `writer-cli` because `writer` is squatted)
cargo install writer-cli

# Homebrew (planned for v0.2)
brew install 199-biotechnologies/tap/writer
```

After install, the binary on your PATH is `writer`.

```bash
writer --version
writer agent-info | jq .
```

## Quick start

```bash
# 1. Create config + data directories and the default voice profile
writer init

# 2. Feed it everything you have ever written
writer learn ~/Documents/drafts/*.md ~/blog/posts/*.md

# 3. Inspect the stylometric fingerprint it computed
writer profile show

# 4. Fine-tune a LoRA adapter on your samples (v0.2)
writer train

# 5. Generate text in your voice (v0.2)
writer write "an essay about why I stopped using Twitter"
writer rewrite draft.md > revised.md
```

`writer learn` and `writer profile show` work today. `writer train`, `writer write`, and `writer rewrite` land in v0.2 -- they return a clear error with a link to the roadmap until then.

## How it works

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   your writing          stylometric          local LoRA             │
│     samples       -->    fingerprint   -->   fine-tune     -->  model
│   (txt, md)           (avg len, SD,         (3B base +         that
│                        vocab, tics)         adapter.sft)      sounds
│                                                               like you
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

1. **Ingest** -- `writer learn` copies your files into `~/.local/share/writer/profiles/<name>/samples/` and tallies word count, character count, sentence length average and standard deviation, and vocabulary size.
2. **Analyse** -- `writer profile show` reports the fingerprint. This is the same kind of metric used to detect AI writing, turned inward.
3. **Train** -- `writer train` runs LoRA fine-tuning on the base model (Llama 3.2 3B by default) using your samples. The output is a small adapter file, a few MB, portable between machines.
4. **Generate** -- `writer write` and `writer rewrite` load the base model plus your adapter and sample from it. Output is yours.

You can have multiple profiles. One for your personal blog, one for work emails, one for fiction. `writer profile use <name>` switches between them.

## Commands

```
writer init                          First-time setup
writer learn <files>                 Ingest writing samples
writer profile show                  Show the active profile's fingerprint
writer profile list                  List all profiles
writer profile new <name>            Create a new profile
writer profile use <name>            Switch active profile
writer train [--profile <name>]      Fine-tune a LoRA adapter
writer write "<prompt>"              Generate in the active voice
writer rewrite <file> [--in-place]   Rewrite a file in the active voice
writer model list                    List available base models
writer model pull <name>             Download a base model
writer config show                   Show effective configuration
writer config path                   Show config file path
writer agent-info                    Machine-readable capability manifest
writer skill install                 Install SKILL.md to agent platforms
writer update [--check]              Self-update from GitHub Releases
```

Every command accepts `--json` (force JSON output) and `--quiet` (suppress human output). Pipe to anything and you get a structured envelope. Run in a terminal and you get colored tables.

## Agent-friendly by default

`writer` is built on the [agent-cli-framework](https://github.com/199-biotechnologies/agent-cli-framework). That means:

| Feature | What you get |
|---|---|
| **`writer agent-info`** | Full JSON manifest of every command, flag, and exit code. Agents call this once to learn the tool. |
| **JSON envelope** | Every command outputs `{"version","status","data"}` when piped. Parse-friendly. |
| **Semantic exit codes** | `0` success, `1` retry, `2` config, `3` input, `4` rate limited. Nothing else. |
| **Errors have suggestions** | Every error includes a `suggestion` field -- a literal instruction the agent can follow. |
| **`writer skill install`** | Writes a `SKILL.md` to `~/.claude/skills/writer/`, `~/.codex/skills/writer/`, and `~/.gemini/skills/writer/` so Claude Code, Codex, and Gemini all discover the tool automatically. |
| **No interactive prompts** | Agents cannot press keys. `writer` never blocks on stdin. |

An AI agent with `writer` on PATH can discover it with `writer agent-info`, learn your voice with `writer learn`, and draft in your voice from then on.

## Privacy

Everything stays local. That is the whole point.

| What | Where | Leaves your machine? |
|---|---|---|
| Writing samples | `~/.local/share/writer/profiles/<name>/samples/` | No |
| Stylometric fingerprint | Computed in-process, returned to you | No |
| Base model weights | `~/.local/share/writer/models/` | Only during initial download from Hugging Face |
| Fine-tuned adapter | `~/.local/share/writer/profiles/<name>/adapter.safetensors` | No |
| Generated text | stdout | No |
| Telemetry | None | -- |

No API keys. No accounts. No upload step. You can run `writer` on an airplane.

## What's inside

```
writer/
├── Cargo.toml
├── src/
│   ├── main.rs            # entry point: parse, detect format, dispatch
│   ├── cli.rs             # clap derive -- all commands and args
│   ├── config.rs          # 3-tier config: defaults < TOML < env vars
│   ├── error.rs           # AppError with exit_code, error_code, suggestion
│   ├── output.rs          # JSON envelope + TTY detection
│   └── commands/
│       ├── init.rs        # directory bootstrap
│       ├── learn.rs       # sample ingestion
│       ├── profile.rs     # voice profile management + stylometric stats
│       ├── train.rs       # LoRA fine-tuning (v0.2)
│       ├── write.rs       # inference (v0.2)
│       ├── rewrite.rs     # inference (v0.2)
│       ├── model.rs       # base model management
│       ├── agent_info.rs  # capability manifest
│       ├── skill.rs       # skill install for agent platforms
│       ├── config.rs      # config show + path
│       └── update.rs      # self-update from GitHub Releases
├── AGENTS.md              # build instructions for AI agents
├── CONTRIBUTING.md
└── LICENSE
```

## Configuration

`~/.config/writer/config.toml`:

```toml
active_profile = "default"
base_model = "llama-3.2-3b-instruct"

[update]
enabled = true
owner = "199-biotechnologies"
repo = "writer"

[inference]
temperature = 0.7
max_tokens = 1024
```

Precedence: compiled defaults < TOML file < env vars (`WRITER_*`) < CLI flags. Env vars use `_` as the separator: `WRITER_INFERENCE_TEMPERATURE=0.9`.

## Roadmap

**v0.1 (shipped)** -- framework scaffolding, profile management, sample ingestion, stylometric analysis, `agent-info`, skill install, self-update.

**v0.2** -- LoRA fine-tuning via MLX on Apple Silicon with llama.cpp fallback, inference for `writer write` and `writer rewrite`, base model downloads.

**v0.3** -- multi-format ingestion (epub, pdf, docx), URL ingestion for blogs, diff scoring (`writer diff original.md generated.md` shows stylometric fidelity), prompt templates.

**v0.4** -- profile merging (blend two voices), adapter export/import for portability, `writer serve` to expose an OpenAI-compatible endpoint for other tools.

## Contributing

Pull requests welcome. Read `AGENTS.md` for the rules (it is short). The short version: keep the framework contract, never add interactive prompts, every error needs a `suggestion`, and `cargo test` must pass.

## A note on the crate name

The `writer` name on crates.io is held by an unmaintained package that has not been updated in over a year and has no repository link. Until that name is reclaimed, this project is published to crates.io as `writer-cli` (the binary is still called `writer`). Installing via `cargo install --git` or Homebrew is unaffected.

## License

MIT. See [LICENSE](LICENSE).

---

<div align="center">

Built by [Boris Djordjevic](https://github.com/longevityboris) at [Paperfoot AI](https://paperfoot.com)

<br />

**If this is useful to you:**

[![Star this repo](https://img.shields.io/github/stars/199-biotechnologies/writer?style=for-the-badge&logo=github&label=%E2%AD%90%20Star%20this%20repo&color=yellow)](https://github.com/199-biotechnologies/writer/stargazers)
&nbsp;&nbsp;
[![Follow @longevityboris](https://img.shields.io/badge/Follow_%40longevityboris-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/longevityboris)

</div>
