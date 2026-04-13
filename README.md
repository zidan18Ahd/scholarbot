---
title: Scholarbot
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# ScholarBot — Research Q&A System

**Live demo:** https://zan18a-scholarbot.hf.space/docs

A production-grade AI system that answers questions using Retrieval Augmented Generation (RAG) + a fine-tuned language model, served via a REST API.

---

## What it does

Send any question → it searches 5000 documents → generates a grounded answer → returns JSON with sources and latency.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | flan-t5-small (google) |
| Retrieval | ChromaDB + sentence-transformers |
| Embeddings | all-MiniLM-L6-v2 |
| API | FastAPI + uvicorn |
| Containerization | Docker |
| Deployment | HuggingFace Spaces |
| CI/CD | GitHub Actions |
| Dataset | natural-questions (5000 rows) |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | /query | Ask a question, get an answer |
| GET | /health | Health check |
| GET | /docs | Swagger UI |

---

## Example Request

```bash
curl -X POST https://zan18a-scholarbot.hf.space/query \
  -H "Content-Type: application/json" \
  -d '{"question": "what is machine learning", "top_k": 3}'
```

## Example Response

```json
{
  "answer": "Machine learning is a process used to learn from data.",
  "sources": ["..."],
  "model_id": "scholarbot-mistral-lora",
  "latency_ms": 652.9
}
```

---

## Local Setup

```bash
git clone https://github.com/zidan18Ahd/scholarbot
cd scholarbot
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python data/download.py
python build_index.py
uvicorn api.main:app --reload --port 8000
```

---

## Author

Built by Zidan Ahmed as an end-to-end AI engineering portfolio project.