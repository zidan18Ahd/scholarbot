from rag.vectorstore import VectorStore

vs = VectorStore()
vs.index_papers('data/raw/arxiv.parquet')