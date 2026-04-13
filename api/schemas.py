from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=512)
    top_k: int = Field(default=3, ge=1, le=10)
    max_new_tokens: int = Field(default=256, le=1024)

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    model_id: str
    latency_ms: float
