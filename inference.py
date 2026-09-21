import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import hf_hub_download

# Load base model + adapter
base_model_name = "sarvamai/sarvam-translate"
adapter_name = "ai4bharat/bhili-translate-bhb-mar"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter_name)
model.eval()

# Load glossary + post-edit module from this repo
glossary_path = hf_hub_download(adapter_name, "glossary.json")
post_edit_path = hf_hub_download(adapter_name, "post_edit.py")

import importlib.util
spec = importlib.util.spec_from_file_location("post_edit", post_edit_path)
pe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pe)

glossary = pe.load_glossary(glossary_path)

# Translate
def translate(text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional Bhili (bhb) to Marathi (mr) translator. "
                "Your goal is to accurately convey the meaning and nuances of the "
                "original Bhili text while adhering to Marathi grammar, vocabulary, "
                "and cultural sensitivities. Produce only the Marathi translation, "
                "without any additional explanations or commentary. Please translate "
                "the following Bhili text into Marathi:"
            ),
        },
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
            early_stopping=True,
        )
    raw = tokenizer.decode(
        output[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True,
    ).strip()
    # Apply glossary post-edit
    edited, _edits = pe.post_edit(
        text, raw, glossary["bhb2mar_sorted_keys"], glossary["bhb2mar"]
    )
    return edited

# Example
print(translate("खेडूतांन वेगीवेळ फवारणी केरनु जोजे हाय."))
print(translate("कोपास्यापापे गुलाबी बोंडअळीखातोर कॉपर ऑक्सिक्लोराईड फवारा."))