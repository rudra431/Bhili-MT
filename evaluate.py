#!/usr/bin/env python3
"""
Bhili NMT Evaluation — chrF++, COMET, and two glossary-based metrics.

Decoding modes (via flags):
  --decoding greedy            : do_sample=False
  --decoding beam              : num_beams=4 + early_stopping
  --use_glossary_postedit      : post-edit predictions to substitute leaked source-language
                                 terms with their glossary target equivalents (only fires
                                 when mar != bhb in glossary)

Metrics:
  - chrF++                   : sacrebleu CHRF (word_order=2)
  - COMET                    : Unbabel/wmt22-comet-da
                               * Reliable for bhb->mar (target language seen by COMET)
                               * UNRELIABLE for mar->bhb (Bhili not in COMET training data)
                                 — score still computed but flagged in report
  - Glossary alignment       : of glossary terms in source, fraction where the prescribed
                               target form appears in prediction. Counts both mar==bhb
                               entries (always satisfied if source term passes through)
                               and mar!=bhb entries. Tells you: "agreement with field
                               team prescription."
  - Translation rate         : restricted to glossary terms where mar != bhb. Of those,
                               fraction where prediction contains the bhb form. This
                               is the cleaner "did the model translate when there's
                               something to translate" signal — most sensitive to the
                               crop-name leakage issue.
"""

import argparse, json, os, re, sys, time
from collections import defaultdict
from datetime import datetime

import torch
from tqdm import tqdm
from sacrebleu.metrics import CHRF
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

DIRECTIONS = {
    "mar2bhb": {"col_src": "mar", "col_tgt": "bhb", "lang_src": "Marathi", "code_src": "mr",
                "lang_tgt": "Bhili", "code_tgt": "bhb", "iso3_src": "mar", "iso3_tgt": "bhb",
                "comet_reliable": False},
    "bhb2mar": {"col_src": "bhb", "col_tgt": "mar", "lang_src": "Bhili", "code_src": "bhb",
                "lang_tgt": "Marathi", "code_tgt": "mr", "iso3_src": "bhb", "iso3_tgt": "mar",
                "comet_reliable": True},
}

SYSTEM_PROMPT = (
    "You are a professional {lang_src} ({code_src}) to {lang_tgt} "
    "({code_tgt}) translator. Your goal is to accurately convey the meaning "
    "and nuances of the original {lang_src} text while adhering to {lang_tgt} "
    "grammar, vocabulary, and cultural sensitivities. Produce only the {lang_tgt} "
    "translation, without any additional explanations or commentary. Please translate "
    "the following {lang_src} text into {lang_tgt}:"
)


# -------------------- glossary helpers --------------------

_DEV_LETTER_RE = re.compile(r"[\u0900-\u097F]")
def _is_dev_letter(ch):
    return bool(_DEV_LETTER_RE.match(ch))


def load_glossary(path):
    with open(path, encoding="utf-8") as f:
        g = json.load(f)
    g["mar2bhb_sorted_keys"] = sorted(g["mar2bhb"].keys(), key=len, reverse=True)
    g["bhb2mar_sorted_keys"] = sorted(g["bhb2mar"].keys(), key=len, reverse=True)
    return g


def _norm_compare(s):
    return re.sub(r"\s+", "", s.lower())


def find_source_terms(text, sorted_keys, lookup):
    """Greedy longest-match scan with word-boundary constraints."""
    found = []
    pos = 0
    n = len(text)
    while pos < n:
        matched = False
        for key in sorted_keys:
            klen = len(key)
            if klen == 0 or pos + klen > n:
                continue
            if text[pos:pos + klen] == key:
                before_ok = (pos == 0) or not _is_dev_letter(text[pos - 1])
                after_ok = (pos + klen == n) or not _is_dev_letter(text[pos + klen])
                if before_ok and after_ok:
                    found.append((key, lookup[key]))
                    pos += klen
                    matched = True
                    break
        if not matched:
            pos += 1
    return found


def _contains_term(text, term):
    """Substring match with left-boundary constraint (lenient on right for inflection)."""
    idx = 0
    while True:
        pos = text.find(term, idx)
        if pos == -1:
            return False
        if pos == 0 or not _is_dev_letter(text[pos - 1]):
            return True
        idx = pos + 1


def _replace_with_boundary(text, old, new):
    """Replace `old` with `new` only at left word-boundaries."""
    out = []
    i = 0
    L = len(old)
    while i < len(text):
        if text[i:i + L] == old:
            left_ok = (i == 0) or not _is_dev_letter(text[i - 1])
            if left_ok:
                out.append(new)
                i += L
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def compute_glossary_metrics(source, prediction, sorted_keys, lookup):
    """Returns dict with:
       - alignment_hits / alignment_total   (all glossary terms)
       - translation_hits / translation_total (only mar != bhb terms)
    """
    src_terms = find_source_terms(source, sorted_keys, lookup)
    align_hits = align_total = trans_hits = trans_total = 0
    for src_t, tgt_t in src_terms:
        align_total += 1
        if _contains_term(prediction, tgt_t):
            align_hits += 1
        if _norm_compare(src_t) != _norm_compare(tgt_t):
            trans_total += 1
            if _contains_term(prediction, tgt_t):
                trans_hits += 1
    return {
        "alignment_hits": align_hits, "alignment_total": align_total,
        "translation_hits": trans_hits, "translation_total": trans_total,
    }


def post_edit(source, prediction, sorted_keys, lookup):
    """Substitute leaked source-form terms with their target equivalents.
       Only fires for entries where mar != bhb (no-op for modern-term entries)."""
    src_terms = find_source_terms(source, sorted_keys, lookup)
    edited = prediction
    edits = []
    for src_t, tgt_t in src_terms:
        if _norm_compare(src_t) == _norm_compare(tgt_t):
            continue  # modern term — no substitution
        if _contains_term(edited, src_t) and not _contains_term(edited, tgt_t):
            new = _replace_with_boundary(edited, src_t, tgt_t)
            if new != edited:
                edits.append((src_t, tgt_t))
                edited = new
    return edited, edits


# -------------------- inference --------------------

def generate_batch(model, tok, prompts, decoding, max_new_tokens):
    inp = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=2048).to(model.device)
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    if decoding == "beam":
        gen_kwargs.update({"num_beams": 4, "early_stopping": True})
    with torch.no_grad():
        out = model.generate(**inp, **gen_kwargs)
    return tok.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)


# -------------------- COMET --------------------

def maybe_load_comet(want_comet):
    if not want_comet:
        return None
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        print("WARN: `unbabel-comet` not installed. Run: pip install unbabel-comet", file=sys.stderr)
        print("WARN: Skipping COMET metric.", file=sys.stderr)
        return None
    print("Loading COMET (Unbabel/wmt22-comet-da)...", file=sys.stderr)
    p = download_model("Unbabel/wmt22-comet-da")
    return load_from_checkpoint(p)


def compute_comet(comet_model, sources, predictions, references):
    if comet_model is None:
        return None
    data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(sources, predictions, references)]
    out = comet_model.predict(data, batch_size=32, gpus=1, progress_bar=False)
    return float(out.system_score)


# -------------------- per-direction eval --------------------

def eval_one(key, args, glossary, comet_model):
    d = DIRECTIONS[key]
    # name = f"bhili-translate-{d['iso3_src']}-{d['iso3_tgt']}"
    # adapter = os.path.join(args.model_base, name)
    adapter = args.model_base
    if not os.path.exists(adapter):
        print(f"  Skip {key} — adapter not found at {adapter}", file=sys.stderr)
        return None

    prompt = SYSTEM_PROMPT.format(
        lang_src=d["lang_src"], code_src=d["code_src"],
        lang_tgt=d["lang_tgt"], code_tgt=d["code_tgt"]
    )
    print(f"\n{'='*70}")
    print(f"  {d['lang_src']} -> {d['lang_tgt']} | {adapter}")
    print(f"  decoding={args.decoding} | postedit={args.use_glossary_postedit}")
    print(f"{'='*70}")

    with open(args.test_path, encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]
    data = [r for r in data
            if r.get(d["col_src"], "").strip() and r.get(d["col_tgt"], "").strip()]
    print(f"  Test samples: {len(data)}")

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()

    if d["col_src"] == "mar":
        sorted_keys, lookup = glossary["mar2bhb_sorted_keys"], glossary["mar2bhb"]
    else:
        sorted_keys, lookup = glossary["bhb2mar_sorted_keys"], glossary["bhb2mar"]

    preds_raw, preds_final, refs, srcs, cats = [], [], [], [], []
    edit_logs = []
    t0 = time.time()
    for i in tqdm(range(0, len(data), args.batch_size), desc=f"  {key} gen"):
        batch = data[i:i + args.batch_size]
        prompts = [
            tok.apply_chat_template(
                [{"role": "system", "content": prompt},
                 {"role": "user", "content": r[d["col_src"]].strip()}],
                tokenize=False, add_generation_prompt=True)
            for r in batch
        ]
        dec = generate_batch(model, tok, prompts, args.decoding, args.max_new_tokens)
        for r, p in zip(batch, dec):
            src = r[d["col_src"]].strip()
            ref = r[d["col_tgt"]].strip()
            cat = r.get("category", "unknown")
            p = p.strip()
            preds_raw.append(p)
            srcs.append(src)
            refs.append(ref)
            cats.append(cat)
            if args.use_glossary_postedit:
                edited, edits = post_edit(src, p, sorted_keys, lookup)
                preds_final.append(edited)
                edit_logs.append(edits)
            else:
                preds_final.append(p)
                edit_logs.append([])
    elapsed_gen = time.time() - t0

    # ----- metrics -----
    chrf_overall = CHRF(word_order=2).corpus_score(preds_final, [refs]).score

    chrf_by_cat = {}
    cat2idx = defaultdict(list)
    for i, c in enumerate(cats):
        cat2idx[c].append(i)
    for c, idxs in cat2idx.items():
        if len(idxs) < 5:
            continue
        sub_p = [preds_final[i] for i in idxs]
        sub_r = [refs[i] for i in idxs]
        chrf_by_cat[c] = {
            "chrf2pp": round(CHRF(word_order=2).corpus_score(sub_p, [sub_r]).score, 4),
            "n": len(idxs),
        }

    align_hits = align_total = trans_hits = trans_total = 0
    per_sample = []
    for s, p in zip(srcs, preds_final):
        m = compute_glossary_metrics(s, p, sorted_keys, lookup)
        align_hits += m["alignment_hits"]
        align_total += m["alignment_total"]
        trans_hits += m["translation_hits"]
        trans_total += m["translation_total"]
        per_sample.append(m)

    alignment = (align_hits / align_total) if align_total > 0 else None
    translation_rate = (trans_hits / trans_total) if trans_total > 0 else None

    comet_score = None
    if comet_model is not None:
        print(f"  Computing COMET...")
        t1 = time.time()
        comet_score = compute_comet(comet_model, srcs, preds_final, refs)
        print(f"  COMET done in {time.time()-t1:.0f}s")

    # ----- save per-sample predictions -----
    pred_path = os.path.join(args.run_dir, f"{key}_predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for s, r, p_raw, p_final, c, edits, ps in zip(
            srcs, refs, preds_raw, preds_final, cats, edit_logs, per_sample
        ):
            f.write(json.dumps({
                "category": c, "source": s, "reference": r,
                "prediction_raw": p_raw, "prediction_final": p_final,
                "post_edits": edits,
                **ps,
            }, ensure_ascii=False) + "\n")

    del model, base
    torch.cuda.empty_cache()

    return {
        "direction": key, "model": adapter, "n_samples": len(data),
        "chrf2pp": round(chrf_overall, 4),
        "chrf2pp_by_category": chrf_by_cat,
        "comet": round(comet_score, 4) if comet_score is not None else None,
        "comet_reliable": d["comet_reliable"],
        "glossary_alignment": round(alignment, 4) if alignment is not None else None,
        "glossary_alignment_hits_total": f"{align_hits}/{align_total}",
        "translation_rate": round(translation_rate, 4) if translation_rate is not None else None,
        "translation_hits_total": f"{trans_hits}/{trans_total}",
        "generation_time_sec": round(elapsed_gen, 1),
    }


# -------------------- main --------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_path", required=True)
    p.add_argument("--model_base", required=True,
                   help="Dir containing bhili-translate-{src}-{tgt} adapter subdirs")
    p.add_argument("--base_model", default="sarvamai/sarvam-translate")
    p.add_argument("--glossary", required=True)
    p.add_argument("--direction", default="all", choices=["all"] + list(DIRECTIONS.keys()))
    p.add_argument("--decoding", default="greedy", choices=["greedy", "beam"])
    p.add_argument("--use_glossary_postedit", action="store_true")
    p.add_argument("--no_comet", action="store_true")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--gpu", type=int, default=None,
                   help="Pin to specific GPU id (sets CUDA_VISIBLE_DEVICES)")
    p.add_argument("--run_name", required=True)
    p.add_argument("--output_dir", default="eval_results")
    args = p.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Pinned to GPU {args.gpu}")

    args.run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.run_dir, exist_ok=True)

    glossary = load_glossary(args.glossary)
    print(f"Loaded glossary: {len(glossary['mar2bhb'])} mar->bhb, "
          f"{len(glossary['bhb2mar'])} bhb->mar")

    comet_model = None if args.no_comet else maybe_load_comet(want_comet=True)

    dirs = list(DIRECTIONS.keys()) if args.direction == "all" else [args.direction]
    results = []
    for dkey in dirs:
        r = eval_one(dkey, args, glossary, comet_model)
        if r:
            results.append(r)

    report = {
        "run_name": args.run_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "model_base": args.model_base, "base_model": args.base_model,
            "decoding": args.decoding,
            "use_glossary_postedit": args.use_glossary_postedit,
            "test_path": args.test_path, "glossary": args.glossary,
        },
        "results": results,
    }
    with open(os.path.join(args.run_dir, "report.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    txt_path = os.path.join(args.run_dir, "report.txt")
    with open(txt_path, "w") as f:
        f.write(f"Bhili NMT Eval | {report['timestamp']} | run={args.run_name}\n")
        f.write(f"Decoding: {args.decoding} | Glossary post-edit: {args.use_glossary_postedit}\n")
        f.write("=" * 110 + "\n")
        f.write(f"{'Direction':<12}{'chrF++':>10}{'COMET':>10}{'Align':>10}{'AlignH/T':>14}"
                f"{'TransRate':>12}{'TransH/T':>12}{'N':>8}\n")
        f.write("-" * 110 + "\n")
        for r in results:
            comet_str = f"{r['comet']:.4f}" if r['comet'] is not None else "-"
            if r['comet'] is not None and not r['comet_reliable']:
                comet_str += "*"
            align_str = f"{r['glossary_alignment']:.4f}" if r['glossary_alignment'] is not None else "-"
            trans_str = f"{r['translation_rate']:.4f}" if r['translation_rate'] is not None else "-"
            f.write(f"{r['direction']:<12}{r['chrf2pp']:>10.4f}{comet_str:>10}"
                    f"{align_str:>10}{r['glossary_alignment_hits_total']:>14}"
                    f"{trans_str:>12}{r['translation_hits_total']:>12}"
                    f"{r['n_samples']:>8}\n")
        f.write("-" * 110 + "\n")
        f.write("* COMET unreliable: Bhili not in COMET training data (use chrF++ and glossary metrics instead)\n")
        f.write("Align     = of glossary terms in source, fraction where prescribed bhb form appears in prediction\n")
        f.write("TransRate = of glossary terms where mar != bhb, fraction where bhb form appears in prediction\n")
        f.write("            (this is the cleaner 'did the model translate when there's something to translate' signal)\n\n")
        for r in results:
            f.write(f"\n[{r['direction']}] chrF++ by category:\n")
            for cat, v in sorted(r['chrf2pp_by_category'].items(),
                                 key=lambda kv: -kv[1]['n']):
                f.write(f"  {cat:<25} chrF++={v['chrf2pp']:>8.4f}  n={v['n']}\n")

    print("\n" + open(txt_path).read())
    print(f"\nFull report: {txt_path}")
    print(f"Predictions: {args.run_dir}/*_predictions.jsonl")


if __name__ == "__main__":
    main()