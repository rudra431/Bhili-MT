# Bhili-MT

Finetuning [sarvam-translate](https://huggingface.co/sarvamai/sarvam-translate) (via LoRA/PEFT) for Marathi ↔ Bhili translation, with a glossary-based post-edit step and an evaluation pipeline built around chrF++, COMET, and two glossary-fidelity metrics.

This README covers the full pipeline end to end: data prep, training, evaluation, inference/testing, and the glossary post-processing step, plus a running list of known data-quality issues found during the September 2026 field-testing review.

## Contents

- [Setup](#setup)
- [Data](#data)
- [1. Clean the data](#1-clean-the-data)
- [2. Split train/eval](#2-split-traineval)
- [3. Train](#3-train)
- [4. Evaluate a checkpoint](#4-evaluate-a-checkpoint)
- [5. Compare runs / plot metrics](#5-compare-runs--plot-metrics)
- [6. Try a checkpoint interactively](#6-try-a-checkpoint-interactively)
- [7. Batch-test specific sentences](#7-batch-test-specific-sentences)
- [Glossary post-processing](#glossary-post-processing)
- [Publishing to the Hugging Face Hub](#publishing-to-the-hugging-face-hub)
- [Known data-quality issues](#known-data-quality-issues)

## Setup

```bash
pip install torch transformers peft trl datasets sacrebleu unbabel-comet gradio --upgrade
```

`unbabel-comet` is only needed for `evaluate.py`'s COMET metric (see caveat below); `gradio` is only needed for `serve_translate.py`. Everything else (training, base evaluation, batch testing, glossary tooling) only needs `torch`/`transformers`/`peft`/`sacrebleu`.

You'll need a GPU with enough memory to hold `sarvam-translate` in bf16 plus a LoRA adapter — training and evaluation both load the full base model. `serve_translate.py` and `batch_test.py` will run on CPU too, just much slower.

## Data

Parallel Bhili/Marathi (and where available Hindi/English) sentence data lives in `data/` as JSONL, one record per line:

```json
{"id": "...", "category": "krishi_darshani", "eng": "...", "hin": "", "mar": "...", "bhb": "..."}
```

`data_nmt_train.jsonl` is the main corpus (36,498 lines); `agri_5k_new_10k.jsonl` (14,843 lines) is an agriculture-focused set that substantially overlaps with it — see [Known data-quality issues](#known-data-quality-issues) before assuming these two files are independent.

`category` groups records by domain (`krishi_darshani`, `general`, `advisory_pop`, `market_price`, `scheme_info`, `livestock`, `weather`, `warehouse_infra`, `numbers_units`, `scheme_grievance`, `bhili_12`, `fact_based`, ...). `general` and `krishi_darshani` dominate the corpus; several agri-specific categories have an order of magnitude fewer examples, which shows up as an accuracy gap in eval too.

Raw TSV exports (e.g. from field-team spreadsheets) can be converted to this JSONL shape with:

```bash
python convert_data.py input.tsv output.jsonl --category advisory_pop
```

(expects TSV columns `dp_id, id, marathi, validated_translation`.)

## 1. Clean the data

`clean.py` normalizes Unicode (NFC), strips invisible characters and stray whitespace, and drops pairs that fail basic sanity checks (too short, too digit-heavy, source/target length ratio too extreme, near-identical source/target implying a copy-through rather than a translation).

```bash
python clean.py \
  --input data/data_nmt_train.jsonl data/agri_5k_new_10k.jsonl \
  --output data/clean.jsonl \
  --rejected data/rejected.jsonl \
  --stats
```

Key thresholds (all overridable): `--min_chars 20`, `--min_words 4`, `--max_ratio 3.5` (max source/target length ratio), `--max_digit_ratio 0.5`. Note the length-ratio audit in [Known data-quality issues](#known-data-quality-issues) below found genuine misalignments at ratios as tight as 2.5–3.2, i.e. inside this filter's default 3.5 cutoff — consider tightening `--max_ratio` if you rerun cleaning, and check `rejected.jsonl` either way.

## 2. Split train/eval

```bash
python split.py --input data/clean.jsonl --test_size 1000 --seed 42
# -> data/clean_train.jsonl, data/clean_test.jsonl
```

**Caveat before you run this**: `split.py` does a flat random shuffle-and-split over individual lines. We found that `id` in the raw data is not a unique per-sentence-pair key — many ids are shared by up to 20 different sentence-pair chunks from the same source document (see [Known data-quality issues](#known-data-quality-issues)). A line-level split can put sibling chunks of the same document on both sides of the train/eval boundary, which is a soft form of train/eval leakage (the eval set won't be cleanly held out, even without exact duplicates). The currently-used eval set was checked and has **zero exact-text overlap** with the training files, so this hasn't bitten a real run yet — but if you regenerate a split from the raw data, group by `id` first, or dedupe/shuffle at the document level rather than the line level.

## 3. Train

```bash
python train.py \
  --data_path data/clean_train.jsonl \
  --output_base output/rudra-bhili-translate \
  --direction all \
  --epochs 7 \
  --lora_r 32 --lora_alpha 64 \
  --batch_size 16 --learning_rate 2e-4
```

- `--direction all` trains both `mar2bhb` and `bhb2mar` adapters back-to-back; pass `--direction bhb2mar` or `--direction mar2bhb` to train just one. Each direction gets its own output folder: `output_base/bhili-translate-{src_iso3}-{tgt_iso3}/` (e.g. `bhili-translate-bhb-mar/`), with checkpoints saved as `checkpoint-<step>` inside it.
- `--targeted_path` optionally points to a smaller, high-priority JSONL (e.g. glossary-term-dense sentences) that gets oversampled `--targeted_oversample` times (default 3x) and mixed into training — use this to upweight categories or terms that are underrepresented in the main corpus.
- `--copy_threshold` (default 0.85) drops pairs where source and target are too character-similar to be a real translation (a safety net independent of `clean.py`'s own filtering).
- `--eval_split` (default 0.05) is a random held-out slice used only for `eval_loss`/early best-checkpoint selection during training — this is separate from, and much smaller than, the standalone eval set used by `evaluate.py`.
- Training uses LoRA (`target_modules="all-linear"`), bf16, cosine LR schedule, and keeps the last 3 checkpoints (`save_total_limit=3`) plus loads the best one (by `eval_loss`) at the end. **Note**: eval_loss is a training-time proxy — the actual best checkpoint by chrF++/glossary metrics can differ (in our review, the *earliest* saved checkpoint of each run scored best on every downstream metric; later checkpoints had drifted). Always confirm with `evaluate.py` on real checkpoints rather than trusting `eval_loss` alone.
- Add `--push_to_hub --hub_org <org>` to push each finished adapter to `<org>/bhili-translate-<src>-<tgt>` on the Hugging Face Hub (private repo).

## 4. Evaluate a checkpoint

```bash
python evaluate.py \
  --test_path data/clean_test.jsonl \
  --model_base output/rudra-bhili-translate/bhili-translate-bhb-mar/checkpoint-19000 \
  --base_model sarvamai/sarvam-translate \
  --glossary glossary.json \
  --direction bhb2mar \
  --decoding beam \
  --use_glossary_postedit \
  --run_name ckp_19000_beam \
  --output_dir output/rudra-bhili-translate/generations/bhb2mar
```

`--model_base` is the path to **one specific adapter checkpoint directory** (not the parent folder) — always pass `--direction` matching that checkpoint's actual training direction; the script doesn't infer it from the path.

Writes `report.json`, `report.txt`, and `<direction>_predictions.jsonl` (per-sample source/reference/`prediction_raw`/`prediction_final`/post-edit log) under `<output_dir>/<run_name>/`. Metrics reported:

- **chrF++** (sacrebleu, `word_order=2`) — the primary quality metric.
- **COMET** (`Unbabel/wmt22-comet-da`) — reliable for `bhb2mar` (Marathi is in COMET's training data); **not reliable for `mar2bhb`**, since Bhili isn't, and the report flags this with a `*`. Pass `--no_comet` to skip it (faster, and avoids an extra heavy model load).
- **Glossary alignment** — of glossary terms found in the source, the fraction where the prescribed target form appears in the prediction.
- **Translation rate** — restricted to glossary terms where the Bhili and Marathi forms actually differ, the fraction where the model produced the translated (not leaked source-language) form. This is the more sensitive "did it actually translate the term" signal.
- `--decoding greedy` (default) or `beam` (`num_beams=4`) — beam decoding scored consistently higher on every metric we checked, at roughly 15% more generation time; recommended for anything beyond a quick sanity check.
- `--use_glossary_postedit` applies `post_edit.py`'s glossary substitution to `prediction_raw` to produce `prediction_final`. **Worth knowing**: as of this writing, several glossary entries are net-harmful (see below) — enabling this flag currently *lowers* chrF++ by roughly 0.6–0.9 points versus leaving it off, across every checkpoint/decoding combination we checked. Don't assume it's a free win; audit the glossary first (see [Glossary post-processing](#glossary-post-processing)).

## 5. Compare runs / plot metrics

```bash
python compare_runs.py output/rudra-bhili-translate/generations/bhb2mar/ckp_19000 \
                        output/rudra-bhili-translate/generations/bhb2mar/ckp_19000_beam \
                        output/rudra-bhili-translate/generations/bhb2mar/ckp_33000
```

Prints a side-by-side table from each run's `report.json`. For trend plots across checkpoints (chrF++, glossary alignment, translation rate, generation time vs. checkpoint step, overall and by category), run `print_eval_metrics.py` — it currently points at `output/rudra-bhili-translate/generations/mar2bhb` via the `ROOT` constant near the top of the file; edit that path (or make it a CLI arg) to point at `bhb2mar` instead, and it writes PNGs to `<ROOT>/plots/`.

## 6. Try a checkpoint interactively

`serve_translate.py` is a small Gradio UI for typing a sentence, picking a direction and decoding strategy, and seeing the raw output, the glossary-post-edited output, which substitutions fired, and (if you paste in a reference) the chrF++ score:

```bash
pip install gradio peft sacrebleu --upgrade
python serve_translate.py
# open http://127.0.0.1:7860
```

By default it loads `checkpoint-19000` for `bhb2mar` (the best-scoring local checkpoint per our eval) and the pushed `ai4bharat/bhili-translate-mar-bhb` Hub adapter for `mar2bhb` (no local checkpoint for that direction was found on disk at review time). Override with `--bhb2mar_ckpt <path>` / `--mar2bhb_adapter <path or hub id>`.

For plain one-off inference without the UI, `inference.py` and `translate.py` are lighter-weight scripts — `inference.py` loads a single hub adapter and glossary post-edit for quick ad hoc checks; `translate.py` runs batch inference over a JSONL file for a given `--src_lang`/`--tgt_lang` pair.

## 7. Batch-test specific sentences

`batch_test.py` runs a checkpoint over a CSV of sentences (one column `bhili_text`) and writes back `prediction_raw`, `prediction_final`, and `glossary_edits` for each:

```bash
python batch_test.py nmt_test_sentences.csv --decoding beam
# -> nmt_test_sentences_results.csv
```

Useful for regression-testing specific known-problem sentences (e.g. pulled from field-call transcripts) against a checkpoint without spinning up the full Gradio UI. `nmt_test_sentences.csv` in this repo has ~25 real sentences pulled from the September 2026 field-testing transcripts, tagged by which reported issue each one probes (crop-entity swap, glossary term confusion, number formatting, garbled/truncated input, etc.) — a reasonable starting set for regression-testing any new checkpoint.

## Glossary post-processing

`glossary.json` (built from the field team's TSV via `clean_glossary.py`) holds two lookup dicts, `bhb2mar` and `mar2bhb`. `post_edit.py` uses it to catch cases where the model's raw translation left a source-language term untranslated (e.g. a Bhili dialect word surviving into the Marathi output) and substitutes the glossary-prescribed target form.

**Two known problem classes, found via field-sentence testing in September 2026:**

1. **Boundary-matching bug (fixed)**: the substitution logic used to be lenient on the right-hand boundary of a match "for inflection," which let short glossary keys match *inside* unrelated compound words — e.g. the entry `बोंड → कवठ` matched the `बोंड` inside `बोंडअळी` ("bollworm") and corrupted it into nonsense `कवठअळी`. `post_edit.py` now requires a real boundary or a recognized Marathi inflectional suffix on the right side too. The previous version is kept as `post_edit.py.bak_pre_boundary_fix` for reference. Empirically this fix is narrow — it touches roughly 1–1.6% of eval sentences and moves chrF++ by well under 0.1 point either direction.

2. **Bad glossary entries (not yet fixed — needs a data edit, not a code edit)**: `audit_glossary.py` replays every glossary substitution that fired across the full eval history (all checkpoints, both directions) and scores whether each one moved predictions closer to or further from the reference:

   ```bash
   pip install sacrebleu
   python audit_glossary.py \
     "output/rudra-bhili-translate/generations/bhb2mar/*/bhb2mar_predictions.jsonl" \
     "output/rudra-bhili-translate/generations/mar2bhb/*/mar2bhb_predictions.jsonl" \
     --output glossary_audit.csv
   ```

   234 distinct entries fired across the eval set; a meaningful number are net-harmful, including several that hurt 100% of the times they fired (`पीएम किसान → प्रधानमंत्री किसान योजना`, `राजधानी → भांडवल`, `उत्पन्न → उत्पादन`, `सिंचन → पाय देवनु`, `तुडतुडे → जासीड`, among others), and a direct contradiction where the glossary has *both* `ग्रॅम → ग्राम` (hurts 78/84 times) and `ग्राम → ग्रॅम` (helps 6/6 times) for the same synonym pair. Full ranked list, worst first, with a concrete example per entry, is in `glossary_audit.csv`. **These haven't been edited in `glossary.json` yet** — that's a deliberate data decision that should probably involve the field/linguistics team, not something to patch silently in code.

## Publishing to the Hugging Face Hub

```bash
python hf_push.py --repos ai4bharat/bhili-translate-bhb-mar ai4bharat/bhili-translate-mar-bhb
```

Uploads `glossary.json` and `post_edit.py` (not the model weights — those go via `train.py --push_to_hub`) to the given Hub repos, so `inference.py`-style loading (`hf_hub_download` for the glossary + post-edit module) works for anyone pulling the adapter from the Hub. Use `--dry_run` first.

## Known data-quality issues

Found during a review of the training corpus and eval predictions in September 2026 (data as of then — re-check before relying on exact counts if the corpus has since changed):

- **`id` is not a unique key.** In `data_nmt_train.jsonl`, 3,618 ids are shared across 2–20 *different* sentence-pair chunks each (9,686 records total) — they're document/paragraph identifiers reused per chunk, not per-pair keys. Any tooling (including earlier versions of the scripts in this repo) that deduplicates or splits by `id` alone will silently drop or mis-group real training examples. The true deduplicated corpus size (removing only exact full-row duplicates) is **46,497 rows**, not the ~36,811 you'd get by naively deduping on `id`.
- **Cross-file duplication.** 4,844 of `agri_5k_new_10k.jsonl`'s 14,843 records are exact duplicates (same id, identical text) of rows already in `data_nmt_train.jsonl`. Concatenating both files as-is for training gives those rows double weight.
- **Sentence/chunk misalignment.** 49 pairs (0.13%, all within `data_nmt_train.jsonl` — 35 `krishi_darshani`, 14 `general`) have a source/target word-count ratio outside a generous 0.4–2.5 band, and inspection confirms these are genuine misalignments rather than natural length variation: either the Bhili and Marathi sides cover different content entirely, or (in the `krishi_darshani` cases) they're chunks of the same document split at different boundaries on each side. Full list with source/target text: `flagged_misaligned_pairs_full.csv`. Plausibly related to the model's tendency (see field-testing notes) to fabricate unrelated content for short/incomplete inputs, since pairs like these teach exactly that pattern during training.
- **Metadata gap**: 9,999 of `agri_5k_new_10k.jsonl`'s 14,843 records (67%) have no `eng` gloss. Doesn't affect bhb↔mar training directly, but limits English-mediated review of that slice.
- **Category imbalance**: `general` (13,624) and `krishi_darshani` (8,344) dominate; several agri-specific categories (`livestock`, `weather`, `market_price`, `scheme_info`, `scheme_grievance`) sit at only ~450–500 examples each, roughly 20x fewer.
- **No exact-text train/eval leakage found** in the eval set currently used by `evaluate.py` — checked directly, 0 of 1,000 eval sources match training text verbatim. The `id`-reuse issue above means this isn't guaranteed to hold for any future split regenerated by `split.py` without grouping by document — see the caveat in [step 2](#2-split-traineval).
