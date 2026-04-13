import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from training.dataset import build_sft_dataset

MODEL_ID = 'mistralai/Mistral-7B-v0.3'

def main():
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_cfg, device_map='auto'
    )
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.05, bias='none',
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    dataset = build_sft_dataset('data/raw/arxiv.parquet')
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        args=SFTConfig(
            output_dir='./outputs',
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy='epoch',
            eval_strategy='epoch',
            report_to='none',
        ),
    )
    trainer.train()
    model.save_pretrained('./lora-adapter')
    tokenizer.save_pretrained('./lora-adapter')
    print('Saved to ./lora-adapter')

if __name__ == '__main__':
    main()
