from rag.vectorstore import VectorStore

class Retriever:
    def __init__(self, store, top_k=3):
        self.store = store
        self.top_k = top_k

    def retrieve(self, query):
        q_embed = self.store.embedder.encode([query]).tolist()
        results = self.store.col.query(
            query_embeddings=q_embed, n_results=self.top_k
        )
        return results['documents'][0]
