from huggingface_hub import HfApi
from transformers import AutoTokenizer
from peft import PeftModel
import os

REPO_ID = os.environ.get('HF_REPO_ID', 'your-username/scholarbot-mistral-lora')

def push():
    api = HfApi()
    api.create_repo(REPO_ID, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained('./lora-adapter')
    tokenizer.push_to_hub(REPO_ID)
    print(f'Live at https://huggingface.co/{REPO_ID}')

if __name__ == '__main__':
    push()
