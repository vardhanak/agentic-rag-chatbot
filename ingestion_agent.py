import io
from multiprocessing import Queue
from typing import Dict, Any
from utils import infer_and_read
from langchain.text_splitter import RecursiveCharacterTextSplitter

class IngestionAgent:
    def __init__(self, in_queue: Queue, out_queue: Queue):
        self.in_q = in_queue
        self.out_q = out_queue
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    def handle_upload(self, msg: Dict[str,Any]):
        files = msg['payload'].get('files', [])
        trace_id = msg.get('trace_id')
        ingest_results = []
        all_chunk_records = []
        print("[IngestionAgent] Received upload request with", len(files), "files")

        for filename, b in files:
            if isinstance(b, (bytes, bytearray)):
                stream = io.BytesIO(b)
            else:
                stream = b
            stream.seek(0)

            doc_id, text = infer_and_read(filename, stream)
            if not text.strip():
                print(f"[IngestionAgent] No text parsed for {filename}")
                continue

            chunks = self.splitter.split_text(text)
            print(f"[IngestionAgent] Parsed {len(chunks)} chunks from {filename}")

            for i, c in enumerate(chunks):
                record = {
                    "doc_id": doc_id,
                    "doc_name": filename,
                    "chunk_id": f"{doc_id}__{i}",
                    "text": c,
                    "meta": {"source": filename, "chunk_index": i},
                }
                all_chunk_records.append(record)

            ingest_results.append({"doc_id": doc_id, "doc_name": filename, "num_chunks": len(chunks)})

        resp = {
            "type": "INGESTION_COMPLETE",
            "sender": "IngestionAgent",
            "trace_id": trace_id,
            "payload": {"results": ingest_results, "chunks": all_chunk_records},
        }
        print("[IngestionAgent] Ingestion complete, sending response with", len(all_chunk_records), "chunks")
        self.out_q.put(resp)

    def run_once(self, msg):
        if msg.get('type') == 'UPLOAD_DOCS':
            self.handle_upload(msg)
