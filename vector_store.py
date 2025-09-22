from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from threading import Lock

class SimpleFAISS:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dim)
        self.metadatas = []
        self.lock = Lock()

    def add(self, texts, metas):
        vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        with self.lock:
            self.index.add(np.array(vecs).astype('float32'))
            for m, t in zip(metas, texts):
                mm = m.copy()
                mm['text'] = t
                self.metadatas.append(mm)

    def search(self, query: str, k: int = 10):
        qvec = self.model.encode([query], convert_to_numpy=True)
        qvec = np.array(qvec).astype('float32')
        with self.lock:
            if self.index.ntotal == 0:
                return []
            D, I = self.index.search(qvec, k)
            results = []
            for idx, dist in zip(I[0], D[0]):
                if 0 <= idx < len(self.metadatas):
                    results.append({'score': float(dist), 'meta': self.metadatas[idx]})
            return results
