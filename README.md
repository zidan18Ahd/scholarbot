# ScholarBot

Fine-tuned Mistral-7B research Q&A system with RAG.

## Stack
- Model: Mistral-7B + QLoRA (peft)
- RAG: ChromaDB + sentence-transformers
- API: FastAPI + uvicorn
- Infra: Docker + GitHub Actions

## Quickstart
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python data/download.py
uvicorn api.main:app --reload
```
