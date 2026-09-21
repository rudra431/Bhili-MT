import os
import json
import argparse
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


LANG_MAP = {
    "eng": "English",
    "hin": "Hindi",
    "mar": "Marathi",
    "tam": "Tamil",
    "tel": "Telugu",
    "kan": "Kannada",
    "mal": "Malayalam",
    "guj": "Gujarati",
    "pan": "Punjabi",
    "ori": "Odia",
    "ben": "Bengali",
    "asm": "Assamese",
    "urd": "Urdu",
    "nep": "Nepali",
    "san": "Sanskrit",
    "snd": "Sindhi",
    "kas": "Kashmiri",
    "kok": "Konkani",
    "mai": "Maithili",
    "mni": "Manipuri",
    "doi": "Dogri",
    "sat": "Santali",
    "bod": "Bodo",
    "bhb": "Bhili"
}


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_prompts(texts, tgt_lang):
    prompts = []

    for text in texts:
        messages = [
            {
                "role": "system",
                "content": f"Translate the text below to {tgt_lang}."
            },
            {
                "role": "user",
                "content": text
            }
        ]

        prompts.append(messages)

    return prompts


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    texts,
    tgt_lang,
    max_new_tokens,
    temperature
):
    messages_batch = build_prompts(texts, tgt_lang)

    formatted_texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in messages_batch
    ]

    model_inputs = tokenizer(
        formatted_texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        num_return_sequences=1
    )

    outputs = []

    for input_ids, generated in zip(model_inputs.input_ids, generated_ids):
        output_ids = generated[len(input_ids):]
        output_text = tokenizer.decode(
            output_ids,
            skip_special_tokens=True
        ).strip()

        outputs.append(output_text)

    return outputs


def get_pending_indices(data, src_lang, tgt_lang):
    indices = []

    for idx, item in enumerate(data):
        src_text = item.get(src_lang, "").strip()
        tgt_text = item.get(tgt_lang, "").strip()

        if src_text and not tgt_text:
            indices.append(idx)

    return indices


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)

    parser.add_argument("--src_lang", type=str, required=True)
    parser.add_argument("--tgt_lang", type=str, required=True)

    parser.add_argument(
        "--model_name",
        type=str,
        default="sarvamai/sarvam-translate"
    )

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.01)

    args = parser.parse_args()

    src_lang_name = LANG_MAP[args.src_lang]
    tgt_lang_name = LANG_MAP[args.tgt_lang]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    data = load_jsonl(args.input_jsonl)

    pending_indices = get_pending_indices(
        data,
        args.src_lang,
        args.tgt_lang
    )

    print(f"Total pending translations: {len(pending_indices)}")

    for start_idx in tqdm(
        range(0, len(pending_indices), args.batch_size)
    ):
        batch_indices = pending_indices[
            start_idx:start_idx + args.batch_size
        ]

        batch_texts = [
            data[idx][args.src_lang].strip()
            for idx in batch_indices
        ]

        translations = generate_batch(
            model=model,
            tokenizer=tokenizer,
            texts=batch_texts,
            tgt_lang=tgt_lang_name,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )

        for idx, translation in zip(batch_indices, translations):
            data[idx][args.tgt_lang] = translation

        save_jsonl(args.output_jsonl, data)

    print("Done")


if __name__ == "__main__":
    main()