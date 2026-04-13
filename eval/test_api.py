from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200

def test_query():
    r = client.post('/query', json={
        'question': 'What is attention in transformers?',
        'top_k': 2,
        'max_new_tokens': 64
    })
    assert r.status_code == 200
    assert len(r.json()['answer']) > 10
