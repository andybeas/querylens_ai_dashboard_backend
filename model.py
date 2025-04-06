import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

class CustomModel:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.data = None

    def train(self, df: pd.DataFrame):
        self.data = df
        sentences = df.fillna("").astype(str).apply(lambda x: ' | '.join(x.str.lower().str.strip()), axis=1).tolist()
        embeddings = self.model.encode(sentences)
        self.index = faiss.IndexFlatL2(len(embeddings[0]))
        self.index.add(embeddings)

    def query(self, text: str, top_k: int = 5):
        if self.index is None or self.data is None:
            raise ValueError("Model is not trained yet. Please upload and train on a CSV.")
        
        embedding = self.model.encode([text])
        D, I = self.index.search(embedding, top_k)
        return self.data.iloc[I[0]].to_dict(orient="records")

    def reset(self):
        self.index = None
        self.data = None
