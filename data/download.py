from datasets import load_dataset
import pathlib

def download_arxiv(max_rows=5_000):
    ds = load_dataset(
        "sentence-transformers/natural-questions",
        split=f"train[:{max_rows}]"
    )
    df = ds.to_pandas()[["query", "answer"]].dropna()
    df.columns = ["article", "abstract"]
    pathlib.Path("data/raw").mkdir(parents=True, exist_ok=True)
    df.to_parquet("data/raw/arxiv.parquet")
    print(f"Saved {len(df)} rows to data/raw/arxiv.parquet")

if __name__ == "__main__":
    download_arxiv()