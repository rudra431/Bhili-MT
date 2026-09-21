#!/usr/bin/env python3

import argparse, json, os, time, torch
from datetime import datetime
from tqdm import tqdm
from sacrebleu.metrics import CHRF
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

DIRECTIONS = {
    "mar2bhb": {"col_src": "mar", "col_tgt": "bhb", "lang_src": "Marathi",  "code_src": "mr",  "lang_tgt": "Bhili",   "code_tgt": "bhb", "iso3_src": "mar", "iso3_tgt": "bhb"},
    "bhb2mar": {"col_src": "bhb", "col_tgt": "mar", "lang_src": "Bhili",   "code_src": "bhb", "lang_tgt": "Marathi",  "code_tgt": "mr",  "iso3_src": "bhb", "iso3_tgt": "mar"},
    "hin2bhb": {"col_src": "hin", "col_tgt": "bhb", "lang_src": "Hindi",   "code_src": "hi",  "lang_tgt": "Bhili",   "code_tgt": "bhb", "iso3_src": "hin", "iso3_tgt": "bhb"},
    "bhb2hin": {"col_src": "bhb", "col_tgt": "hin", "lang_src": "Bhili",   "code_src": "bhb", "lang_tgt": "Hindi",   "code_tgt": "hi",  "iso3_src": "bhb", "iso3_tgt": "hin"},
    "eng2bhb": {"col_src": "eng", "col_tgt": "bhb", "lang_src": "English", "code_src": "en",  "lang_tgt": "Bhili",   "code_tgt": "bhb", "iso3_src": "eng", "iso3_tgt": "bhb"},
    "bhb2eng": {"col_src": "bhb", "col_tgt": "eng", "lang_src": "Bhili",   "code_src": "bhb", "lang_tgt": "English", "code_tgt": "en",  "iso3_src": "bhb", "iso3_tgt": "eng"},
}

SYSTEM_PROMPT = (
    "You are a professional {lang_src} ({code_src}) to {lang_tgt} "
    "({code_tgt}) translator. Your goal is to accurately convey the meaning "
    "and nuances of the original {lang_src} text while adhering to {lang_tgt} "
    "grammar, vocabulary, and cultural sensitivities. Produce only the {lang_tgt} "
    "translation, without any additional explanations or commentary. Please translate "
    "the following {lang_src} text into {lang_tgt}:"
)


def eval_one(key, args):
    d = DIRECTIONS[key]
    name = f"bhili-translate-{d['iso3_src']}-{d['iso3_tgt']}"
    adapter = os.path.join(args.model_base, name)
    if not os.path.exists(adapter): print(f"  Skip {key} — {adapter} not found"); return None

    prompt = SYSTEM_PROMPT.format(lang_src=d["lang_src"], code_src=d["code_src"], lang_tgt=d["lang_tgt"], code_tgt=d["code_tgt"])
    print(f"\n{'='*60}\n  {d['lang_src']} → {d['lang_tgt']} | {name}\n{'='*60}")

    with open(args.test_path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]
    data = [r for r in data if r.get(d["col_src"], "").strip() and r.get(d["col_tgt"], "").strip()]
    print(f"  Test: {len(data)}")

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True), adapter)
    model.eval()

    preds, refs, recs = [], [], []
    t0 = time.time()
    for i in tqdm(range(0, len(data), args.batch_size), desc=f"  {key}"):
        batch = data[i:i+args.batch_size]
        prompts = [tok.apply_chat_template([{"role":"system","content":prompt},{"role":"user","content":r[d["col_src"]].strip()}], tokenize=False, add_generation_prompt=True) for r in batch]
        batch_refs = [r[d["col_tgt"]].strip() for r in batch]
        inp = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad(): out = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False)
        dec = tok.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)
        for p, r, item in zip(dec, batch_refs, batch):
            p = p.strip(); preds.append(p); refs.append(r)
            recs.append({"source": item[d["col_src"]], "reference": r, "prediction": p})

    elapsed = time.time() - t0
    score = CHRF(word_order=2).corpus_score(preds, [refs])
    pred_path = os.path.join(args.output_dir, f"{key}_predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  chrF2++: {score.score:.4f} | {elapsed:.0f}s")
    del model; torch.cuda.empty_cache()
    return {"direction": key, "model": name, "chrf2pp": round(score.score, 4), "samples": len(data), "time": round(elapsed, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_path", required=True)
    p.add_argument("--model_base", required=True)
    p.add_argument("--base_model", default="sarvamai/sarvam-translate")
    p.add_argument("--direction", default="all", choices=["all"]+list(DIRECTIONS.keys()))
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--output_dir", default="eval_results")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dirs = list(DIRECTIONS.keys()) if args.direction == "all" else [args.direction]

    results = [r for d in dirs if (r := eval_one(d, args))]

    rpt = os.path.join(args.output_dir, "eval_report.txt")
    with open(rpt, "w") as f:
        f.write(f"Bhili NMT Eval | {datetime.now():%Y-%m-%d %H:%M} | {args.model_base}\n{'='*70}\n")
        f.write(f"{'Direction':<15}{'Model':<30}{'chrF2++':>10}{'N':>8}\n{'-'*63}\n")
        for r in results: f.write(f"{r['direction']:<15}{r['model']:<30}{r['chrf2pp']:>10.4f}{r['samples']:>8}\n")
        if results: f.write(f"{'-'*63}\n{'Average':<45}{sum(r['chrf2pp'] for r in results)/len(results):>10.4f}\n")

    print(f"\n{'='*60}\n{'Direction':<15}{'chrF2++':>10}\n{'-'*25}")
    for r in results: print(f"{r['direction']:<15}{r['chrf2pp']:>10.4f}")
    if results: print(f"{'-'*25}\n{'Average':<15}{sum(r['chrf2pp'] for r in results)/len(results):>10.4f}")
    print(f"\nReport: {rpt}")

if __name__ == "__main__": main()