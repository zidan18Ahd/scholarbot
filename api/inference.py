import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ADAPTER_PATH = './lora-adapter'
BASE_MODEL   = 'mistralai/Mistral-7B-v0.3'

PROMPT = '### Question: {question}\n### Context: {context}\n### Answer:'

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map='auto'
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    return model, tokenizer

def generate(model, tokenizer, question, context, max_new_tokens=256):
    prompt = PROMPT.format(question=question, context=context[:1024])
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return decoded.split('### Answer:')[-1].strip()
