# Session Handoff — Phases 0-5 complete, Gemma 4 26B running, Codex review in

**Date:** 2026-04-09 23:30
**Session:** Full implementation sprint — Phases 0 through 5, end-to-end generation on Gemma 4 26B
**Context usage at handoff:** Very high — full plan + implementation + benchmarking + research

---

## Active Plan

**Plan file:** `docs/superpowers/plans/2026-04-09-writer-quality-architecture.md` (2,405 lines)
**Plan status:** Phases 0-5 implemented. Phase 3.5 (benchmark harness) added per handoff spec. Phases 6-9 remain.

| Phase | Status |
|---|---|
| 0: Foundation | **Done** — tagged `phase0-complete` |
| 1: Data pipeline | **Done** — sources, normalize, dedupe, chunk, ingest |
| 2: Stylometric fingerprint v2 | **Done** — 9 evidence-based features |
| 3: Ollama inference backend | **Done** — /api/chat, MLX, multi-candidate |
| 3.5: Benchmark harness | **Done** — writer-bench binary |
| 4: Quality decoding layer | **Done** — logit bias, rank-N, filter, priming |
| 5: mlx-tune LoRA backend | **Wired** — training data prepared, HF model path needs fix |
| 6: DPO against AI-rewrites | Not started |
| 7: Contrastive decoding (CoPe) | Not started |
| 8: `writer score` CLI | Not started |
| 9: Extensibility docs | Not started |

---

## What Was Accomplished This Session

- **Phase 0** (10 tasks): All backend traits, types, configs. 14 tests. Tagged `phase0-complete`.
- **Phase 1**: Source parsers (markdown, plain text, Obsidian vault), normalize chain (HTML strip, SVG strip, pandoc divs, link anchors, image refs, escape chars, signatures, zero-width, tracking params), dedupe, chunker, ingest pipeline. 8 tests.
- **Phase 2**: 9-category stylometric fingerprint grounded in research: lengths, function words, char n-grams, punctuation, readability (Flesch-Kincaid/Coleman-Liau/ARI), vocabulary richness (Yule's K/hapax legomena/Simpson's D), AI-slop detection (70 words + 30 phrases). 10 tests.
- **Phase 3**: Ollama HTTP backend via `/api/chat` endpoint (required for Gemma 4 thinking model). Multi-candidate generation. Wiremock tests. 4 tests.
- **Phase 3.5**: `writer-bench` binary with VoiceFidelity + SlopScore + CreativeRubric → combined score.
- **Phase 4**: Quality decoding layer — generate N candidates, rank by stylometric distance, logit bias (suppress AI slop, boost user vocab), post-hoc filter, system prompt priming.
- **Phase 5**: MlxTuneBackend wired, training data preparation (441 train + 49 valid Adams samples in mlx-lm chat format). HF model path needs fixing before actual training run.
- **Ollama upgraded** from 0.17.6 to 0.20.4 (MLX backend, flash attention).
- **Gemma 4 26B** (17GB) pulled and generating. Confirmed working via `/api/chat`.
- **Douglas Adams corpus ingested**: 492 samples, 684,089 words, 22,653 vocabulary.
- **Benchmarks captured**:
  - Held-out Adams: distance 0.121, combined 0.883
  - Gemma 3 4B generation: distance 0.282
  - Gemma 4 26B generation: distance 0.297 (single sample 0.297, needs tuning)
- **Codex adversarial review completed** — found 8 issues (see Gotchas section).
- **Research agent completed** — LUAR, Binoculars, PAN tasks, missing features identified.

### Files created/modified this session

All source files under `src/`:
- `src/backends/` — types.rs, inference/ (mod, capabilities, request, response, ollama), training/ (mod, config, artefact, mlx_tune)
- `src/bench/` — mod.rs, combined.rs, slop_score.rs, voice_fidelity.rs
- `src/bin/writer_bench.rs`
- `src/corpus/` — mod.rs, sample.rs, sources/ (mod, markdown, plain_text, obsidian), normalize.rs, dedupe.rs, chunker.rs, ingest.rs
- `src/decoding/` — mod.rs, logit_bias.rs, ranker.rs, filter.rs, prompts.rs
- `src/stylometry/` — mod.rs, fingerprint.rs, scoring.rs, ai_slop.rs, features/ (mod, lengths, function_words, ngrams, punctuation, vocabulary, readability, richness)
- `src/lib.rs`, `src/main.rs`, `src/config.rs`, `src/output.rs`
- `src/commands/` — write.rs, rewrite.rs, model.rs, learn.rs, train.rs, init.rs, agent_info.rs
- `tests/` — phase0_types.rs, phase0_inference_backend.rs, phase0_training_backend.rs, phase0_sample.rs, phase1_sources.rs, phase2_stylometry.rs, phase3_ollama.rs
- `benchmarks/` — baseline-2026-04-09.json, gemma4-baseline-2026-04-09.json
- `README.md`, `Cargo.toml`

---

## Key Decisions Made

1. **Gemma 4 26B requires `/api/chat`, not `/api/generate`.** The generate endpoint returns empty `response` field. Gemma 4 is a "thinking model" — it uses a `thinking` field for chain-of-thought and `content` for the final answer. Must use `num_predict >= 4096` to give enough tokens for thinking + response.

2. **System prompt priming is double-edged.** Prescriptive priming ("avg sentence length: 6.8 words") makes the model produce choppy, conformist text. The improved version uses softer guidance ("use short punchy sentences mixed with occasional longer ones"). Still needs tuning.

3. **Multi-candidate generation on 26B is slow.** 8 candidates = ~3-4 minutes. Reduced to 2 candidates for practical use. Consider async/streaming in future, or a draft model for fast candidates + one 26B verification pass.

4. **Normalize chain removes stylometric signal** (Codex finding). `normalize_whitespace()` and `normalize_quotes()` flatten spacing and quote style which are part of the author's fingerprint. Need to either: (a) run fingerprinting BEFORE normalization, or (b) preserve stylometric-relevant whitespace/quote patterns.

5. **AI-slop penalty uses substring matching, not occurrence counting** (Codex finding). One use of "delve" and twenty uses score the same. Should count actual occurrences, not just presence.

6. **Training data format:** mlx-lm expects `{train,valid,test}.jsonl` with chat format: `{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}`. The user prompt is generic ("Write a {context} passage in your natural voice.") — this may need refinement for better SFT.

7. **Douglas Adams sentence length is low (6.8 mean)** because `unicode_segmentation::unicode_sentences()` splits on ALL periods including abbreviations. This inflates sentence count and deflates mean. Consider a more robust sentence splitter or calibrate expectations.

---

## Current State

- **Branch:** `main`
- **Last commit:** `5dcc076 bench: Gemma 4 26B baseline — distance 0.297 on M4 Max MLX`
- **Uncommitted changes:** None (clean working tree)
- **Tests passing:** Yes — 36+ tests across 7 test files
- **Build status:** Clean (warnings only, no errors)
- **Remote:** https://github.com/199-biotechnologies/writer — all pushed
- **Ollama:** 0.20.4 running with gemma4:26b (17GB), gemma3:4b (3.3GB), qwen3.5:9b (6.6GB)
- **Config:** `base_model = "google/gemma-4-26b"`, `n_candidates = 2`, `max_tokens = 4096`
- **Corpus:** Adams 492 samples in `/Users/biobook/Library/Application Support/writer/profiles/default/samples/corpus.jsonl`
- **Fingerprint:** Saved at `/Users/biobook/Library/Application Support/writer/profiles/default/fingerprint.json`
- **Training data:** Prepared at `/Users/biobook/Library/Application Support/writer/profiles/default/training_data/{train,valid,test}.jsonl`

---

## What to Do Next

1. **Read this handoff.** Every section.

2. **Fix Codex review issues (priority order):**
   - **HIGH: SVG stripping bug** in `src/corpus/normalize.rs` — inline `<svg>...</svg>` on one line gets stuck in `in_svg` state. Fix: handle single-line SVG blocks.
   - **HIGH: candidate_index metadata wrong** in `src/decoding/mod.rs:82` — candidates arrive out-of-order from JoinSet, index doesn't match backend index.
   - **MEDIUM: normalization removes signal** — run fingerprinting before normalization, or preserve quote/whitespace patterns.
   - **MEDIUM: slop penalty uses presence not count** — change to count occurrences, not just `contains()`.
   - **MEDIUM: tests are weak** — `scoring_human_text_closer_to_human_fingerprint` tests against the SAME text it fingerprinted (proves nothing about generalization). Add held-out test samples.

3. **Fix HuggingFace model path for `writer train`.** The current config sends `google/gemma-4-26b` but mlx-lm needs the HuggingFace repo path. Check: `python3 -c "from huggingface_hub import model_info; print(model_info('google/gemma-4-26b-a4b'))"` to find the correct path. May need `google/gemma-4-26b-a4b` or `mlx-community/gemma-4-26b-a4b` for MLX weights.

4. **Run LoRA training:**
   ```bash
   writer train --json
   ```
   If HF path is fixed, this will train for 1000 steps on 441 Adams samples. Expected: ~30 minutes on M4 Max, loss should drop from ~3.0 to ~1.5.

5. **Re-benchmark with LoRA adapter.** Expected distance drop from 0.297 to ~0.15-0.20.

6. **Start autoresearch loop:**
   ```bash
   autoresearch start \
     --metric "writer-bench run --json | jq -r '.data.combined_score'" \
     --baseline-command "writer-bench baseline --save" \
     --improvement-threshold 0.01 \
     --max-iterations 50 \
     --eval-timeout 900 \
     --keep-on-regression false
   ```

7. **Phase 6: DPO against AI-rewrites.** Generate AI rewrites of Adams samples using base model → build preference dataset → train DPO adapter.

8. **Phase 7: Contrastive decoding.** Check if Ollama 0.20.4 supports logprobs. If not, use `mlx_lm.generate` Python escape hatch behind `WRITER_ALLOW_PYTHON=1`.

---

## Codex Adversarial Review (GPT-5.4, full findings)

| Severity | Issue | Location |
|---|---|---|
| HIGH | SVG stripping: inline `<svg>...</svg>` on one line leaves `in_svg` stuck | `src/corpus/normalize.rs` strip_svg_blocks |
| HIGH | `candidate_index` lies when candidates complete out-of-order from JoinSet | `src/decoding/mod.rs:82` |
| MEDIUM | `GenerationEvent::Error` is fatal — one bad candidate kills all | `src/decoding/mod.rs:113` |
| MEDIUM | Normalization destroys stylometric signal (whitespace, quotes) | `src/corpus/normalize.rs:179,191,213` |
| MEDIUM | AI-slop penalty counts presence, not occurrences | `src/stylometry/scoring.rs:249,256` |
| MEDIUM | Test `sentence_lengths_differ_between_styles` is vacuous | `tests/phase2_stylometry.rs:30` |
| MEDIUM | Test `scoring_human_text_closer_to_human_fingerprint` proves nothing | `tests/phase2_stylometry.rs:93` |
| LOW | `fingerprint_serializes_roundtrip` only checks word_count | `tests/phase2_stylometry.rs:84` |
| OK | Combined score formula matches spec | `src/bench/combined.rs:28` |

---

## Research Findings (from background agent)

Repos to integrate:
- **LLNL/LUAR** (github.com/LLNL/LUAR) — transformer authorship embeddings, Apache-2.0
- **ahans30/Binoculars** (github.com/ahans30/Binoculars) — zero-shot AI detection, ICML 2024

Missing features (evidence-based, should add):
- POS tag n-grams (needs spaCy or lightweight tagger)
- Discourse markers frequency
- Sentence-initial word patterns
- LUAR embedding cosine similarity as scoring dimension

Key papers:
- "Layered Insights" (EMNLP 2025) — hybrid interpretable + neural outperforms either alone
- CoPe (EMNLP 2025) — contrastive decoding for personalized LLMs
- "Can LLMs Identify Authorship?" (EMNLP 2024 Findings)

---

## Files to Review First

1. `docs/superpowers/plans/2026-04-09-writer-quality-architecture.md` — the master plan
2. `src/decoding/mod.rs` — the quality pipeline (FIX: candidate_index + error handling)
3. `src/corpus/normalize.rs` — the cleanup chain (FIX: SVG + signal preservation)
4. `src/stylometry/scoring.rs` — distance function (FIX: slop penalty counting)
5. `src/backends/inference/ollama.rs` — Ollama backend (/api/chat, thinking model handling)
6. `benchmarks/gemma4-baseline-2026-04-09.json` — current numbers to beat

---

## Gotchas & Warnings

1. **Gemma 4 26B is a thinking model.** Empty `content` field is normal if `num_predict` is too low — the model uses all tokens for thinking. Set `max_tokens >= 4096`.

2. **`/api/chat` not `/api/generate`.** The Ollama backend was switched to use the chat endpoint. Old wiremock test was updated. If reverting, all generation breaks.

3. **Ollama was upgraded from 0.17.6 to 0.20.4.** If Ollama regresses after a brew update, the old version is at `/opt/homebrew/Cellar/ollama/0.17.6.reinstall/`. The new one is at `/usr/local/bin/ollama`.

4. **Three duplicate `ollama pull` processes were killed.** Only one should run. Check `ps aux | grep "ollama pull"` before starting new pulls.

5. **First sample in corpus.jsonl has HTML artifacts.** Sample 0 is the cover page from epub conversion. The normalize pipeline handles most artifacts but the first file's initial content may have residual SVG (per Codex finding).

6. **`writer train` will fail** until the HuggingFace model path is correct. The model name `google/gemma-4-26b` is the writer internal ID, not the HF repo. Check `ollama show gemma4:26b` for the actual model family, then find the HF path.

7. **Single-agent sessions: do NOT use TaskCreate/TaskUpdate.** Global CLAUDE.md rule.

8. **Douglas Adams corpus is at** `/Users/biobook/Projects/douglas-adamiser/DA/markdown/` (9 books, 538K words raw). Already ingested into default profile.

9. **User's Obsidian vaults** are at `/Users/biobook/Library/Mobile Documents/iCloud~md~obsidian/Documents/` — the 199 vault has 968K words, not yet ingested. User wants to support multiple voice profiles (Douglas Adams + their own voice).

10. **The README was updated** but still references some v0.1 language. The config section now shows the correct Gemma 4 defaults but may need another pass after LoRA training works.

---

## Commands to Run After Loading This Handoff

```bash
cd /Users/biobook/Projects/writer
git status --short
git log --oneline -5
cargo build --release 2>&1 | tail -5
ollama list
./target/release/writer --version
./target/release/writer write "Test sentence." --json 2>&1 | head -5
./target/release/writer-bench run --smoke --json 2>&1
```

Expected: clean tree, Gemma 4 26B in model list, generation working, benchmark producing combined score ~0.88 on smoke test.
