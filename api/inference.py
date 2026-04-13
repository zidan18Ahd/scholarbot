from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_ID = "google/flan-t5-small"

PROMPT = "Answer this question: {question}\nContext: {context}\nAnswer:"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    model.eval()
    return model, tokenizer

def generate(model, tokenizer, question, context, max_new_tokens=256):
    prompt = PROMPT.format(question=question, context=context[:512])
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0], skip_special_tokens=True)