#!/usr/bin/env python3
"""
Batch-test the bhb2mar checkpoint on a CSV of real sentences (pulled from the
field-testing call transcripts) and write out the translations for review.

Usage:
    pip install peft --upgrade   # torch/transformers/sacrebleu already needed for serve_translate.py
    python batch_test.py nmt_test_sentences.csv

    # override the checkpoint (default: ckp_19000, your best-scoring one)
    python batch_test.py nmt_test_sentences.csv --bhb2mar_ckpt output/rudra-bhili-translate/bhili-translate-bhb-mar/checkpoint-33000

Input CSV must have a column named "bhili_text" (the sentences to translate);
any other columns (issue_id, session_tab, timestamp, english_gloss,
what_to_check, ...) are passed through untouched.

Output: <input>_results.csv with two new columns added:
    prediction_raw       -- model output before glossary post-edit
    prediction_final     -- model output after glossary post-edit
    glossary_edits       -- which substitutions fired, if any
"""

import argparse
import csv

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import post_edit as pe

BASE_MODEL = "sarvamai/sarvam-translate"
GLOSSARY_PATH = "glossary.json"

DEFAULT_ADAPTER = "output/rudra-bhili-translate/bhili-translate-bhb-mar/checkpoint-19000"
SYSTEM_PROMPT = (
    "You are a professional Bhili (bhb) to Marathi (mr) translator. "
    "Your goal is to accurately convey the meaning and nuances of the "
    "original Bhili text while adhering to Marathi grammar, vocabulary, "
    "and cultural sensitivities. Produce only the Marathi translation, "
    "without any additional explanations or commentary. Please translate "
    "the following Bhili text into Marathi:"
)


def load_model(adapter, decoding):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model


def translate_one(tokenizer, model, text, decoding, glossary):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    gen_kwargs = dict(max_new_tokens=256, early_stopping=True, do_sample=False)
    gen_kwargs["num_beams"] = 4 if decoding == "beam" else 1

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    raw = tokenizer.decode(
        output[0][len(inputs.input_ids[0]):], skip_special_tokens=True
    ).strip()

    sorted_keys = glossary["bhb2mar_sorted_keys"]
    lookup = glossary["bhb2mar"]
    edited, edits = pe.post_edit(text, raw, sorted_keys, lookup)
    edits_str = "; ".join(f"{s}->{t}" for s, t in edits)

    return raw, edited, edits_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("--bhb2mar_ckpt", default=DEFAULT_ADAPTER)
    parser.add_argument("--decoding", choices=["beam", "greedy"], default="beam")
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    out_path = args.output_csv or args.input_csv.rsplit(".", 1)[0] + "_results.csv"

    glossary = pe.load_glossary(GLOSSARY_PATH)

    print(f"Loading base model + adapter from {args.bhb2mar_ckpt} ...")
    tokenizer, model = load_model(args.bhb2mar_ckpt, args.decoding)

    with open(args.input_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fieldnames = list(rows[0].keys()) + ["prediction_raw", "prediction_final", "glossary_edits"]

    with open(out_path, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows):
            text = row["bhili_text"]
            print(f"[{i+1}/{len(rows)}] {text[:60]}...")
            raw, edited, edits_str = translate_one(
                tokenizer, model, text, args.decoding, glossary
            )
            row["prediction_raw"] = raw
            row["prediction_final"] = edited
            row["glossary_edits"] = edits_str
            writer.writerow(row)
            out_f.flush()

    print(f"\nDone. Results written to {out_path}")


if __name__ == "__main__":
    main()
