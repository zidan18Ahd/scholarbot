import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd

EMBED_MODEL = 'all-MiniLM-L6-v2'

class VectorStore:
    def __init__(self, persist_dir='./chroma_db'):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.col = self.client.get_or_create_collection('papers')
        self.embedder = SentenceTransformer(EMBED_MODEL)

    def index_papers(self, parquet_path, batch_size=256):
        df = pd.read_parquet(parquet_path)
        texts = df['abstract'].tolist()
        ids = [str(i) for i in range(len(texts))]
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeds = self.embedder.encode(batch).tolist()
            self.col.add(documents=batch, embeddings=embeds,
                         ids=ids[i:i+batch_size])
        print(f'Indexed {len(texts)} documents')
