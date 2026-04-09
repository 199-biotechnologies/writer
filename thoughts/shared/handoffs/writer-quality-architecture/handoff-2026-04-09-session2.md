# Session Handoff — writer Phase 4 complete, ready for LoRA + autoresearch

**Date:** 2026-04-09 (night session)
**Session:** Phases 0-4 implemented, benchmarked, Douglas Adams corpus ingested
**Previous handoff:** `thoughts/shared/handoffs/writer-quality-architecture/handoff-2026-04-09.md`

---

## What Was Accomplished This Session

1. **Phase 0 complete** — Foundation types, traits, backends (14 tests, tagged `phase0-complete`)
2. **Phase 1 complete** — Data pipeline: markdown, plain text, Obsidian source parsers + normalize (strip HTML, SVG, pandoc divs, link anchors, image refs, escape chars, signatures, zero-width, tracking params) + dedupe + chunker + ingest pipeline (8 tests)
3. **Phase 2 complete** — Stylometric fingerprint v2 with 9 evidence-based feature categories: lengths, function words, char n-grams, punctuation, readability (Flesch-Kincaid, Coleman-Liau, ARI), vocabulary richness (Yule's K, hapax legomena, Simpson's D), AI-slop detection (70 banned words + 30 phrases). Grounded in: PAN shared tasks, Writeprints (Abbasi & Chen 2008), Yule 1944, Simpson 1949, EMNLP 2025 "Layered Insights" (10 tests)
4. **Phase 3 complete** — Ollama HTTP backend with streaming, multi-candidate generation, model tag translation (4 tests via wiremock)
5. **Phase 3.5 complete** — `writer-bench` binary: VoiceFidelity + SlopScore + CreativeRubric → combined score
6. **Phase 4 complete** — Quality decoding layer: generate 8 candidates, rank by stylometric distance, logit bias (suppress AI slop, boost user vocab), post-hoc filter, system prompt priming from fingerprint
7. **Douglas Adams corpus ingested** — 492 samples, 684,089 words, 22,653 vocabulary. Fingerprint saved.
8. **Ollama upgraded to 0.20.4** (MLX backend, flash attention, KV cache quant)
9. **README updated** with Gemma 4, evidence-based stylometry, quality stack, paper references
10. **Codex review dispatched** — waiting for adversarial review results

---

## Douglas Adams Stylometric Fingerprint (key stats)

- Sentence length: mean 6.8, SD 4.3 (short, punchy sentences)
- Word length: mean 4.44 (simple words)
- Flesch-Kincaid grade: 3.94 (very readable — characteristic Adams)
- Questions per 1k: 8.95 (lots of rhetorical questions)
- Exclamations per 1k: 5.24 (expressive)
- Em-dashes per 1k: 1.28 (moderate use)
- Yule's K: 85.33 (rich vocabulary)
- Hapax legomena: 24.7% (many unique words)
- AI-slop banned words not in Adams: 42 (ironic number)
- Preferred words: "arthur", "zaphod", "romana", "something", "think", "looked"

---

## Benchmark Baseline (saved in `benchmarks/baseline-2026-04-09.json`)

| Metric | Held-out Adams | Generated (gemma3:4b) |
|---|---|---|
| Voice fidelity distance | 0.121 | 0.282 |
| Slop density | 0.044% | not yet scored |
| Combined score | 0.883 | not yet computed |
| N samples | 50 | 10 |
| Model | — | gemma3:4b |
| Decoding | — | 8 candidates, logit bias, priming |

**Gap to close:** 0.282 → 0.121 (generation should match held-out human)

---

## Current State

- **Branch:** `main`
- **Last commit:** `136e755 bench: save baseline`
- **Tests:** all passing (36+ tests across 5 test files)
- **Build:** clean release
- **Remote:** pushed to https://github.com/199-biotechnologies/writer
- **Ollama:** 0.20.4 running with gemma3:4b loaded
- **Gemma 4 26B:** pull may still be in progress — check with `ollama list`

---

## What to Do Next

### Immediate (close the gap)

1. **Check if Gemma 4 26B finished pulling** — `ollama list`. If yes, update config `base_model = "google/gemma-4-26b"` and re-run the generation benchmark. Expect significant quality improvement from the larger model.

2. **Read the Codex review** — check for critical issues flagged in the adversarial review. Fix any CRITICAL/HIGH items before proceeding.

3. **Phase 5: LoRA training via mlx-tune**
   - Install mlx-tune: `pip install mlx-lm` (or `uvx mlx-tune`)
   - Implement `MlxTuneBackend` in `src/backends/training/mlx_tune.rs`
   - Convert corpus JSONL to mlx-lm chat format
   - Wire `writer train` to spawn `mlx_lm.lora --train`
   - Train on the Douglas Adams corpus
   - Expected: distance drops from 0.282 to ~0.15-0.20

4. **Phase 6: DPO against AI rewrites**
   - Generate AI rewrites of Adams samples (rejected column)
   - Build preference dataset: {chosen: original Adams, rejected: AI rewrite}
   - Train DPO adapter on top of LoRA SFT
   - Expected: distance drops further to ~0.10-0.15

5. **Start autoresearch loop:**
   ```bash
   autoresearch start \
     --metric "writer-bench run --json | jq -r '.data.combined_score'" \
     --baseline-command "writer-bench baseline --save" \
     --improvement-threshold 0.01 \
     --max-iterations 50 \
     --eval-timeout 900 \
     --keep-on-regression false
   ```

### Research findings to implement

From the stylometry research agent:
- **LUAR embeddings** (github.com/LLNL/LUAR) — neural authorship vectors, add as scoring dimension
- **Binoculars** (github.com/ahans30/Binoculars) — zero-shot AI detection, ICML 2024
- **POS tag n-grams** — strongest syntactic discriminator, needs spaCy or lightweight tagger
- **Discourse markers** — frequency of "however", "therefore", "actually"
- **Sentence-initial patterns** — distribution of sentence starters

### Other items requested by user

- Prepare crates.io publish (`cargo publish` as `writer-cli`)
- Prepare Homebrew formula
- Test with user's Obsidian vault (199 vault, 968K words)
- Support multiple voices/profiles (Douglas Adams, user's own voice, etc.)
- GitHub optimization (the README was updated but check SEO/topics)

---

## Gotchas

1. **Model tag translation:** `google/gemma-4-26b` maps to `gemma4:26b` in Ollama. The translation logic in `src/backends/inference/ollama.rs:model_id_to_ollama_tag()` strips hyphens between model family and version, puts colon before size. Test with `ollama list` to verify the actual tag name.

2. **First sample in corpus has HTML:** Sample 0 in corpus.jsonl may still contain cover page SVG from epub conversion. The normalize pipeline runs AFTER source parsing, and the first file's cover page passes through because it's a single block. Consider pre-filtering samples with word count < 20 or running a second pass cleanup.

3. **Sentence length stats seem low (6.8 mean):** This is because unicode_segmentation's `unicode_sentences()` splits on periods including abbreviations like "Mr." and "St." — inflating sentence count. Consider switching to a more robust sentence splitter or calibrating the expected baseline.

4. **Combined score of 0.883 on held-out is misleading:** This is high because slop score and voice fidelity are both excellent for human text. The generation combined score will be lower because the model's output has higher distance AND may have slop words.

5. **CreativeRubric is a placeholder (72.0):** This needs the EQ-Bench subset prompts + Claude Sonnet 4.6 as judge. Until then, the creative component contributes a fixed 0.144 to the combined score.

---

## Files Modified This Session

**New files:**
- `src/decoding/mod.rs`, `logit_bias.rs`, `ranker.rs`, `filter.rs`, `prompts.rs`
- `src/bench/mod.rs`, `combined.rs`, `slop_score.rs`, `voice_fidelity.rs`
- `src/bin/writer_bench.rs`
- `src/stylometry/features/readability.rs`, `richness.rs`
- `src/corpus/sources/markdown.rs`, `plain_text.rs`, `obsidian.rs`
- `src/corpus/normalize.rs`, `dedupe.rs`, `chunker.rs`, `ingest.rs`, `sources/mod.rs`
- `benchmarks/baseline-2026-04-09.json`
- All test files: `tests/phase0_*.rs`, `tests/phase1_sources.rs`, `tests/phase2_stylometry.rs`, `tests/phase3_ollama.rs`

**Modified:**
- `Cargo.toml` — added deps: tokio-stream, shellexpand; added writer-bench binary; added lib section
- `src/lib.rs` — exposed all modules
- `src/main.rs` — async main, uses writer_cli:: imports
- `src/commands/write.rs`, `rewrite.rs`, `model.rs`, `learn.rs` — wired to real backends
- `src/config.rs` — added decoding + training config sections
- `README.md` — updated with Gemma 4, quality stack, paper references
