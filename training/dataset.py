from datasets import Dataset
import pandas as pd

PROMPT = '### Question: {question}\n### Context: {context}\n### Answer:'

def build_sft_dataset(parquet_path):
    df = pd.read_parquet(parquet_path)
    records = [
        {'text': PROMPT.format(
            question='Summarise the key contributions.',
            context=row['abstract'][:512]
        ) + ' ' + row['abstract'][:256] + ' '}
        for _, row in df.iterrows()
    ]
    return Dataset.from_list(records).train_test_split(test_size=0.05)
