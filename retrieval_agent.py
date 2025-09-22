from multiprocessing import Queue
from typing import Dict, Any, List
from sentence_transformers import CrossEncoder
from vector_store import SimpleFAISS
from datetime import datetime
import time

def _log(msg):
    print(f"[{datetime.now().isoformat()}] [RetrievalAgent] {msg}")

class RetrievalAgent:
    def __init__(self, in_q: Queue, out_q: Queue,
                 K_RETRIEVE=50, K_RERANK=10, rerank_model='cross-encoder/ms-marco-MiniLM-L-6-v2',
                 embedding_model='all-MiniLM-L6-v2'):
        self.in_q = in_q
        self.out_q = out_q
        self.K_RETRIEVE = K_RETRIEVE
        self.K_RERANK = K_RERANK
        _log("Initializing SimpleFAISS and (maybe) reranker...")
        self.vs = SimpleFAISS(model_name=embedding_model)
        _log(f"Loaded embedding model (dim={self.vs.dim}).")
        self.reranker = None
        if K_RERANK and rerank_model:
            _log(f"Loading reranker model: {rerank_model}")
            self.reranker = CrossEncoder(rerank_model)
            _log("Reranker loaded.")

    def handle_chunks_add(self, chunks: List[Dict[str,Any]]):
        _log(f"Received CHUNKS_ADD with {len(chunks)} chunks — starting indexing")
        start = time.time()
        texts = [c['text'] for c in chunks]
        metas = []
        for c in chunks:
            meta = {
                'doc_id': c.get('doc_id'),
                'chunk_id': c.get('chunk_id'),
                'doc_name': c.get('doc_name'),
                'source': c.get('meta', {}).get('source'),
                'chunk_index': c.get('meta', {}).get('chunk_index'),
                'text': c.get('text') 
            }
            metas.append(meta)

        self.vs.add(texts, metas)
        elapsed = time.time() - start
        total = getattr(self.vs.index, "ntotal", "unknown")
        _log(f"Indexed {len(chunks)} chunks in {elapsed:.2f}s. Total vectors now: {total}")

    def do_retrieval(self, msg: Dict[str,Any]):
        query = msg['payload']['query']
        trace = msg.get('trace_id')
        _log(f"RETRIEVAL_REQUEST trace={trace} q='{query[:120]}' — searching top {self.K_RETRIEVE}")
        start = time.time()
        raw_results = self.vs.search(query, k=self.K_RETRIEVE)
        elapsed = time.time() - start
        _log(f"FAISS search returned {len(raw_results)} candidates in {elapsed:.2f}s")
        candidates = []
        for r in raw_results:
            meta = r['meta']
            text = meta.get('text', '')
            candidates.append({'text': text, 'meta': meta, 'score': r['score']})
        final = candidates
        if self.reranker and len(candidates) > 0:
            _log(f"Reranking top {min(len(candidates), self.K_RERANK)} candidates with CrossEncoder")
            top_for_rerank = candidates[:self.K_RERANK]
            pairs = [[query, c['text']] for c in top_for_rerank]
            t0 = time.time()
            scores = self.reranker.predict(pairs)
            t1 = time.time()
            _log(f"Reranker scored {len(scores)} pairs in {t1-t0:.2f}s")
            for c, s in zip(top_for_rerank, scores):
                c['rerank_score'] = float(s)
            top_for_rerank.sort(key=lambda x: x['rerank_score'], reverse=True)
            final = top_for_rerank
        else:
            final = candidates[:self.K_RERANK]
        top_chunks = [{'text': c['text'], 'meta': c['meta'], 'score': c.get('rerank_score', c.get('score'))} for c in final]
        _log(f"Returning {len(top_chunks)} top chunks to LLMResponseAgent (trace={trace})")
        resp = {
            'type': 'RETRIEVAL_RESULT',
            'sender': 'RetrievalAgent',
            'receiver': 'LLMResponseAgent',
            'trace_id': trace,
            'payload': {'retrieved_context': top_chunks, 'query': query}
        }
        self.out_q.put(resp)

    def run_once(self, msg):
        t = msg.get('type')
        if t == 'CHUNKS_ADD':
            self.handle_chunks_add(msg['payload'].get('chunks', []))
        elif t == 'RETRIEVAL_REQUEST':
            self.do_retrieval(msg)
