import time
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from api.schemas import QueryRequest, QueryResponse
from api.inference import load_model, generate
from rag.vectorstore import VectorStore
from rag.retriever import Retriever

store = retriever = model = tokenizer = None

@asynccontextmanager
async def lifespan(app):
    global store, retriever, model, tokenizer
    store = VectorStore()
    retriever = Retriever(store)
    model, tokenizer = load_model()
    yield

app = FastAPI(title='ScholarBot API', version='1.0.0', lifespan=lifespan)

@app.post('/query', response_model=QueryResponse)
async def query(req: QueryRequest):
    t0 = time.perf_counter()
    try:
        docs = retriever.retrieve(req.question)
        context = '\n'.join(docs)
        answer = generate(model, tokenizer, req.question,
                          context, req.max_new_tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return QueryResponse(
        answer=answer, sources=docs,
        model_id='scholarbot-mistral-lora',
        latency_ms=round((time.perf_counter()-t0)*1000, 1)
    )

@app.get('/health')
async def health():
    return {'status': 'ok'}
