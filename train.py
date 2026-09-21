#!/usr/bin/env python3

import argparse, json, os, random, re, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

DIRECTIONS = {
    "mar2bhb": {"col_src": "mar", "col_tgt": "bhb", "lang_src": "Marathi",  "code_src": "mr",  "lang_tgt": "Bhili",   "code_tgt": "bhb", "iso3_src": "mar", "iso3_tgt": "bhb"},
    "bhb2mar": {"col_src": "bhb", "col_tgt": "mar", "lang_src": "Bhili",   "code_src": "bhb", "lang_tgt": "Marathi",  "code_tgt": "mr",  "iso3_src": "bhb", "iso3_tgt": "mar"},
    # "hin2bhb": {"col_src": "hin", "col_tgt": "bhb", "lang_src": "Hindi",   "code_src": "hi",  "lang_tgt": "Bhili",   "code_tgt": "bhb", "iso3_src": "hin", "iso3_tgt": "bhb"},
    # "bhb2hin": {"col_src": "bhb", "col_tgt": "hin", "lang_src": "Bhili",   "code_src": "bhb", "lang_tgt": "Hindi",   "code_tgt": "hi",  "iso3_src": "bhb", "iso3_tgt": "hin"},
    # "eng2bhb": {"col_src": "eng", "col_tgt": "bhb", "lang_src": "English", "code_src": "en",  "lang_tgt": "Bhili",   "code_tgt": "bhb", "iso3_src": "eng", "iso3_tgt": "bhb"},
    # "bhb2eng": {"col_src": "bhb", "col_tgt": "eng", "lang_src": "Bhili",   "code_src": "bhb", "lang_tgt": "English", "code_tgt": "en",  "iso3_src": "bhb", "iso3_tgt": "eng"},
}

SYSTEM_PROMPT = (
    "You are a professional {lang_src} ({code_src}) to {lang_tgt} "
    "({code_tgt}) translator. Your goal is to accurately convey the meaning "
    "and nuances of the original {lang_src} text while adhering to {lang_tgt} "
    "grammar, vocabulary, and cultural sensitivities. Produce only the {lang_tgt} "
    "translation, without any additional explanations or commentary. Please translate "
    "the following {lang_src} text into {lang_tgt}:"
)

TEST = {
    "mar": ["शेतकऱ्यांना सकाळी फवारणी करणे आवश्यक आहे.", "कापसावरील गुलाबी बोंडअळीसाठी कॉपर ऑक्सिक्लोराईड फवारा."],
    "hin": ["किसानों को सुबह छिड़काव करना जरूरी है।", "कपास पर गुलाबी बॉलवर्म के लिए कॉपर ऑक्सीक्लोराइड का छिड़काव करें।"],
    "bhb": ["खेडूतांन वेगीवेळ फवारणी केरनु जोजे हाय.", "कोपास्यापापे गुलाबी बोंडअळीखातोर कॉपर ऑक्सिक्लोराईड फवारा."],
    "eng": ["Farmers need to spray in the morning.", "Spray Copper Oxychloride for Pink Bollworm on cotton."],
}


def load_data(path, src, tgt, threshold):
    with open(path, "r", encoding="utf-8") as f:
        raw = [json.loads(l) for l in f if l.strip()]
    clean = []
    for r in raw:
        s, t = r.get(src, "").strip(), r.get(tgt, "").strip()
        if not s or not t: continue
        a, b = re.sub(r"\s+", "", s.lower()), re.sub(r"\s+", "", t.lower())
        if a and b and sum(1 for x, y in zip(a, b) if x == y) / max(len(a), len(b)) >= threshold: continue
        clean.append(r)
    print(f"  Data: {len(clean)}/{len(raw)} valid pairs")
    return clean


def train_one(key, args):
    d = DIRECTIONS[key]
    name = f"bhili-translate-{d['iso3_src']}-{d['iso3_tgt']}"
    out = os.path.join(args.output_base, name)
    prompt = SYSTEM_PROMPT.format(lang_src=d["lang_src"], code_src=d["code_src"], lang_tgt=d["lang_tgt"], code_tgt=d["code_tgt"])

    print(f"\n{'='*60}\n  {d['lang_src']} → {d['lang_tgt']} | {name}\n{'='*60}")

    data = load_data(args.data_path, d["col_src"], d["col_tgt"], args.copy_threshold)
    if args.targeted_path and os.path.exists(args.targeted_path):
        targeted = load_data(args.targeted_path, d["col_src"], d["col_tgt"], 1.0)
        print(f"  Targeted: {len(targeted)} x {args.targeted_oversample}")
        for _ in range(args.targeted_oversample): data.extend(targeted)
    random.seed(42); random.shuffle(data)

    split = Dataset.from_list(data).train_test_split(test_size=args.eval_split, seed=42)
    print(f"  Train: {len(split['train'])} | Eval: {len(split['test'])}")

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tok.pad_token = tok.eos_token; tok.padding_side = "right"

    def fmt(ex):
        return {"text": [tok.apply_chat_template([
            {"role": "system", "content": prompt},
            {"role": "user", "content": ex[d["col_src"]][i]},
            {"role": "assistant", "content": ex[d["col_tgt"]][i]},
        ], tokenize=False, add_generation_prompt=False) for i in range(len(ex[d["col_src"]]))]}

    train_ds = split["train"].map(fmt, batched=True, remove_columns=split["train"].column_names)
    eval_ds = split["test"].map(fmt, batched=True, remove_columns=split["test"].column_names)

    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False; model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules="all-linear"))
    model.print_trainable_parameters()

    SFTTrainer(
        model=model, train_dataset=train_ds, eval_dataset=eval_ds, processing_class=tok,
        args=TrainingArguments(
            output_dir=out, num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum, learning_rate=args.learning_rate,
            warmup_ratio=0.05, weight_decay=0.01, lr_scheduler_type="cosine",
            logging_steps=args.logging_steps, save_strategy="steps", save_steps=args.save_steps,
            eval_strategy="steps", eval_steps=args.eval_steps, save_total_limit=3,
            load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
            bf16=True, report_to="none",
        ),
    ).train()

    model.save_pretrained(out); tok.save_pretrained(out)
    if args.push_to_hub and args.hub_org:
        model.push_to_hub(f"{args.hub_org}/{name}", private=True)
        tok.push_to_hub(f"{args.hub_org}/{name}", private=True)

    print(f"\nValidation:")
    model.eval()
    for s in TEST.get(d["col_src"], TEST["eng"]):
        inp = tok([tok.apply_chat_template([{"role":"system","content":prompt},{"role":"user","content":s}], tokenize=False, add_generation_prompt=True)], return_tensors="pt").to(model.device)
        with torch.no_grad(): o = model.generate(**inp, max_new_tokens=256, do_sample=True, temperature=0.01)
        print(f"  IN:  {s}\n  OUT: {tok.decode(o[0][len(inp.input_ids[0]):], skip_special_tokens=True).strip()}\n")

    del model; torch.cuda.empty_cache()
    print(f"✓ {name} done\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--targeted_path", default=None)
    p.add_argument("--output_base", required=True)
    p.add_argument("--eval_split", type=float, default=0.05)
    p.add_argument("--direction", default="all", choices=["all"]+list(DIRECTIONS.keys()))
    p.add_argument("--base_model", default="sarvamai/sarvam-translate")
    p.add_argument("--epochs", type=int, default=7)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--copy_threshold", type=float, default=0.85)
    p.add_argument("--targeted_oversample", type=int, default=3)
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_org", default=None)
    args = p.parse_args()
    os.makedirs(args.output_base, exist_ok=True)
    dirs = list(DIRECTIONS.keys()) if args.direction == "all" else [args.direction]
    print(f"Model: {args.base_model} | Epochs: {args.epochs} | LoRA: r={args.lora_r} | Targeted: {args.targeted_oversample}x")
    print(f"Directions: {', '.join(dirs)}\n")
    for d in dirs: train_one(d, args)
    print(f"\n{'='*60}\nAll complete!\n{'='*60}")

if __name__ == "__main__": main()