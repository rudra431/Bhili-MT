#!/usr/bin/env python3
"""
Interactive UI to spot-check Bhili <-> Marathi translation quality.

Loads the sarvam-translate base model + a PEFT LoRA adapter for one direction,
lets you type a sentence, pick greedy or beam decoding, and shows:
  - the raw model output
  - the output after the same glossary post-edit used in evaluate.py
  - which glossary substitutions fired (if any)
  - chrF++ against an optional reference you paste in

Run this in the same environment you used for train.py / evaluate.py
(it needs torch + a GPU to be fast, though it will run on CPU too, just slower).

Usage:
    pip install gradio peft sacrebleu --upgrade
    python serve_translate.py
    # then open the printed http://127.0.0.1:7860 URL

    # to pick a specific bhb2mar checkpoint instead of the default (ckp_19000,
    # the best-scoring one per output/rudra-bhili-translate/generations):
    python serve_translate.py --bhb2mar_ckpt output/rudra-bhili-translate/bhili-translate-bhb-mar/checkpoint-33000
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sacrebleu.metrics import CHRF

import post_edit as pe

BASE_MODEL = "sarvamai/sarvam-translate"
GLOSSARY_PATH = "glossary.json"

DIRECTIONS = {
    "bhb2mar": {
        "display": "Bhili -> Marathi",
        # best checkpoint per your eval (66.95 chrF++, beam) — override with --bhb2mar_ckpt
        "adapter": "output/rudra-bhili-translate/bhili-translate-bhb-mar/checkpoint-19000",
        "system_prompt": (
            "You are a professional Bhili (bhb) to Marathi (mr) translator. "
            "Your goal is to accurately convey the meaning and nuances of the "
            "original Bhili text while adhering to Marathi grammar, vocabulary, "
            "and cultural sensitivities. Produce only the Marathi translation, "
            "without any additional explanations or commentary. Please translate "
            "the following Bhili text into Marathi:"
        ),
        "glossary_key": "bhb2mar",
    },
    "mar2bhb": {
        "display": "Marathi -> Bhili",
        # no local checkpoint saved for this direction — falls back to the pushed adapter
        "adapter": "ai4bharat/bhili-translate-mar-bhb",
        "system_prompt": (
            "You are a professional Marathi (mr) to Bhili (bhb) translator. "
            "Your goal is to accurately convey the meaning and nuances of the "
            "original Marathi text while adhering to Bhili grammar, vocabulary, "
            "and cultural sensitivities. Produce only the Bhili translation, "
            "without any additional explanations or commentary. Please translate "
            "the following Marathi text into Bhili:"
        ),
        "glossary_key": "mar2bhb",
    },
}

_loaded = {}  # direction -> (tokenizer, model)
_glossary = pe.load_glossary(GLOSSARY_PATH)


def get_model(direction):
    if direction in _loaded:
        return _loaded[direction]

    cfg = DIRECTIONS[direction]
    adapter = cfg["adapter"]
    print(f"[{direction}] loading base model + adapter from {adapter} ...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    _loaded[direction] = (tokenizer, model)
    return tokenizer, model


def translate(direction, text, decoding, reference):
    text = (text or "").strip()
    if not text:
        return "", "", "(enter a sentence)", ""

    cfg = DIRECTIONS[direction]
    tokenizer, model = get_model(direction)

    messages = [
        {"role": "system", "content": cfg["system_prompt"]},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    gen_kwargs = dict(max_new_tokens=256, early_stopping=True, do_sample=False)
    if decoding == "beam":
        gen_kwargs["num_beams"] = 4
    else:
        gen_kwargs["num_beams"] = 1

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    raw = tokenizer.decode(
        output[0][len(inputs.input_ids[0]):], skip_special_tokens=True
    ).strip()

    sorted_keys = _glossary[f"{cfg['glossary_key']}_sorted_keys"]
    lookup = _glossary[cfg["glossary_key"]]
    edited, edits = pe.post_edit(text, raw, sorted_keys, lookup)

    edits_str = (
        "\n".join(f"{s}  ->  {t}" for s, t in edits)
        if edits
        else "(no glossary substitutions fired)"
    )

    score_str = ""
    reference = (reference or "").strip()
    if reference:
        chrf = CHRF(word_order=2).sentence_score(edited, [reference]).score
        score_str = f"chrF++ (post-edited vs reference): {chrf:.2f}"

    return raw, edited, edits_str, score_str


def build_ui():
    import gradio as gr

    with gr.Blocks(title="Bhili <-> Marathi translation check") as demo:
        gr.Markdown("## Bhili <-> Marathi translation quality check")
        direction = gr.Radio(
            choices=[(cfg["display"], key) for key, cfg in DIRECTIONS.items()],
            value="bhb2mar",
            label="Direction",
        )
        decoding = gr.Radio(
            choices=["beam", "greedy"], value="beam", label="Decoding"
        )
        text_in = gr.Textbox(label="Sentence to translate", lines=3)
        reference = gr.Textbox(
            label="Reference translation (optional, for chrF++)", lines=2
        )
        btn = gr.Button("Translate", variant="primary")

        raw_out = gr.Textbox(label="Raw model output", lines=3)
        edited_out = gr.Textbox(label="After glossary post-edit", lines=3)
        edits_out = gr.Textbox(label="Glossary substitutions applied", lines=4)
        score_out = gr.Textbox(label="Score")

        btn.click(
            translate,
            inputs=[direction, text_in, decoding, reference],
            outputs=[raw_out, edited_out, edits_out, score_out],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="get a public gradio.live link")
    parser.add_argument(
        "--bhb2mar_ckpt",
        default=None,
        help="override the bhb2mar checkpoint dir (default: checkpoint-19000)",
    )
    parser.add_argument(
        "--mar2bhb_adapter",
        default=None,
        help="override the mar2bhb adapter (local dir or HF repo id)",
    )
    args = parser.parse_args()

    if args.bhb2mar_ckpt:
        DIRECTIONS["bhb2mar"]["adapter"] = args.bhb2mar_ckpt
    if args.mar2bhb_adapter:
        DIRECTIONS["mar2bhb"]["adapter"] = args.mar2bhb_adapter

    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)
