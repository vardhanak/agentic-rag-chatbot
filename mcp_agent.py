import threading
import time
from multiprocessing import Queue, Process
from datetime import datetime
import signal
import sys
import uuid
import agent_processes as ap

_LOG_PREFIX = "[MCP Broker]"

def _log(msg: str):
    print(f"[{datetime.now().isoformat()}] {_LOG_PREFIX} {msg}", flush=True)

class MCPBroker:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", K_RETRIEVE=50, K_RERANK=10, llm_model="llama3.2:1b"):
        self.ing_out_public = Queue()
        self.ret_out_public = Queue()
        self.llm_out_public = Queue()

        try:
            self.ing_out_public._mcp_public = True
            self.ret_out_public._mcp_public = True
            self.llm_out_public._mcp_public = True
        except Exception:
            pass

        self.ing_in = Queue()
        self.ret_in = Queue()
        self.llm_in = Queue()

        self.ing_out_internal = Queue()
        self.ret_out_internal = Queue()
        self.llm_out_internal = Queue()

        self._stop_event = threading.Event()
        self._broker_thread = None
        self._procs = []
        self.embedding_model = embedding_model
        self.K_RETRIEVE = K_RETRIEVE
        self.K_RERANK = K_RERANK
        self.llm_model = llm_model

        self._in_queues = {
            "IngestionAgent": self.ing_in,
            "RetrievalAgent": self.ret_in,
            "LLMResponseAgent": self.llm_in,
        }
        self._internal_outs = {
            "IngestionAgent": self.ing_out_internal,
            "RetrievalAgent": self.ret_out_internal,
            "LLMResponseAgent": self.llm_out_internal,
        }
        self._public_outs = {
            "IngestionAgent": self.ing_out_public,
            "RetrievalAgent": self.ret_out_public,
            "LLMResponseAgent": self.llm_out_public,
        }

        self._method_lock = threading.Lock()

    def start(self):
        _log("Starting agent processes...")

        p_ing = Process(target=ap.run_ingestion_agent, args=(self.ing_in, self.ing_out_internal), daemon=True)
        p_ret = Process(
            target=ap.run_retrieval_agent,
            args=(self.ret_in, self.ret_out_internal, self.K_RETRIEVE, self.K_RERANK, self.embedding_model),
            daemon=True,
        )
        p_llm = Process(target=ap.run_llm_agent, args=(self.llm_in, self.llm_out_internal, self.llm_model), daemon=True)

        p_ing.start(); _log(f"IngestionAgent started (pid={p_ing.pid})")
        p_ret.start(); _log(f"RetrievalAgent started (pid={p_ret.pid})")
        p_llm.start(); _log(f"LLMResponseAgent started (pid={p_llm.pid})")

        self._procs = [p_ing, p_ret, p_llm]

        self._broker_thread = threading.Thread(target=self._broker_loop, daemon=True)
        self._broker_thread.start()
        _log("Broker thread started.")

        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception:
            pass

    def _signal_handler(self, signum, frame):
        _log(f"Received signal {signum}, stopping MCPBroker...")
        self.stop()
        sys.exit(0)

    def _broker_loop(self):
        internal_map = self._internal_outs
        public_map = self._public_outs
        in_map = self._in_queues

        entries = list(internal_map.items())
        poll_sleep = 0.05
        _log("Entering broker loop. Routing messages between agents.")

        while not self._stop_event.is_set():
            any_msg = False
            for agent_name, q in entries:
                try:
                    msg = q.get(timeout=0.01)
                except Exception:
                    continue
                any_msg = True

                try:
                    public_q = public_map.get(agent_name)
                    if public_q:
                        try:
                            public_q.put(dict(msg))
                        except Exception:
                            public_q.put(msg)
                except Exception as e:
                    _log(f"Failed to put into public out queue for {agent_name}: {e}")

                try:
                    receiver = msg.get("receiver")
                    if receiver in in_map:
                        try:
                            in_map[receiver].put(msg)
                            _log(f"Routed msg type={msg.get('type')} trace={msg.get('trace_id')} from {msg.get('sender')} -> {receiver}")
                        except Exception as e:
                            _log(f"Failed to route msg to {receiver}: {e}")
                    else:
                        _log(f"Public message from {msg.get('sender')} to {receiver} (type={msg.get('type')})")

                    if msg.get('type') == 'INGESTION_COMPLETE' and isinstance(msg.get('payload', {}).get('chunks'), list):
                        chunks = msg['payload']['chunks']
                        if chunks:
                            forward_msg = {
                                'type': 'CHUNKS_ADD',
                                'sender': msg.get('sender', 'MCPBroker'),
                                'receiver': 'RetrievalAgent',
                                'trace_id': msg.get('trace_id'),
                                'payload': {'chunks': chunks},
                            }
                            try:
                                in_map['RetrievalAgent'].put(forward_msg)
                                _log(f"Auto-forwarded {len(chunks)} chunks to RetrievalAgent (trace={msg.get('trace_id')})")
                            except Exception as e:
                                _log(f"Failed to auto-forward chunks to RetrievalAgent: {e}")
                        
                    if msg.get('type') == 'RETRIEVAL_COMPLETE':
                        top_chunks = msg['payload']['retrieved_context']
                        query = msg['payload']['query']
                        trace = msg.get('trace_id')
                        if chunks:
                            forward_msg = {
                                'type': 'RETRIEVAL_RESULT',
                                'sender': 'MCPBroker',
                                'receiver': 'LLMResponseAgent',
                                'trace_id': trace,
                                'payload': {'retrieved_context': top_chunks, 'query': query}
                            }
                            try:
                                in_map['LLMResponseAgent'].put(forward_msg)
                                _log(f"Auto-forwarded {len(chunks)} chunks to LLMResponseAgent (trace={msg.get('trace_id')})")
                            except Exception as e:
                                _log(f"Failed to auto-forward chunks to LLMResponseAgent: {e}")
                except Exception as e:
                    _log(f"Error during routing: {e}")

            if not any_msg:
                time.sleep(poll_sleep)

        _log("Broker loop exiting.")

    def stop(self, terminate_procs=True):
        _log("Stopping MCPBroker...")
        self._stop_event.set()
        if self._broker_thread:
            self._broker_thread.join(timeout=2.0)

        if terminate_procs:
            for p in self._procs:
                try:
                    if p.is_alive():
                        p.terminate()
                        _log(f"Terminated process pid={p.pid}")
                except Exception:
                    pass
        _log("Stopped.")

    def get_queues_for_coordinator(self):
        try:
            self.ing_out_public._mcp_public = True
            self.ret_out_public._mcp_public = True
            self.llm_out_public._mcp_public = True
        except Exception:
            pass

        return {
            "ing_in": self.ing_in, "ing_out": self.ing_out_public,
            "ret_in": self.ret_in, "ret_out": self.ret_out_public,
            "llm_in": self.llm_in, "llm_out": self.llm_out_public,
        }

    def upload_files(self, files, timeout=None):
        with self._method_lock:
            trace_id = f"upload-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"
            msg = {
                "type": "UPLOAD_DOCS",
                "sender": "MCPBroker",
                "receiver": "IngestionAgent",
                "trace_id": trace_id,
                "payload": {"files": files},
            }
            _log(f"Posting UPLOAD_DOCS trace={trace_id} files={len(files)}")
            self.ing_in.put(msg)
            start = time.time()
            while True:
                if timeout is not None and (time.time() - start) > timeout:
                    raise TimeoutError("upload_files timed out waiting for INGESTION_COMPLETE")
                try:
                    resp = self.ing_out_public.get(timeout=0.5)
                    _log(f"Received from ingestion: {resp.get('type')} trace={resp.get('trace_id')}")
                    if resp.get("type") == "INGESTION_COMPLETE" and resp.get("trace_id") == trace_id:
                        chunks = resp["payload"].get("chunks", [])
                        _log(f"INGESTION_COMPLETE: got {len(chunks)} chunks (trace={trace_id})")
                        return resp
                    else:
                        _log(f"Ignoring ingestion message type={resp.get('type')} trace={resp.get('trace_id')}")
                except Exception:
                    continue

    def ask_query(self, query, timeout=None):
        with self._method_lock:
            trace_id = f"query-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"
            msg = {
                "type": "RETRIEVAL_REQUEST",
                "sender": "MCPBroker",
                "receiver": "RetrievalAgent",
                "trace_id": trace_id,
                "payload": {"query": query},
            }
            _log(f"Posting RETRIEVAL_REQUEST trace={trace_id} q='{query[:120]}'")
            self.ret_in.put(msg)

            start = time.time()
            _log("Waiting for LLM_ANSWER on llm_out...")
            while True:
                if timeout is not None and (time.time() - start) > timeout:
                    raise TimeoutError("ask_query timed out waiting for LLM_ANSWER")

                try:
                    llm_msg = self.llm_out_public.get(timeout=0.5)
                    _log(f"Got message from llm_out: type={llm_msg.get('type')} trace={llm_msg.get('trace_id')}")
                    if llm_msg.get("type") == "LLM_ANSWER" and llm_msg.get("trace_id") == trace_id:
                        _log("Received matching LLM_ANSWER -> returning to UI")
                        return llm_msg
                    else:
                        _log(f"Ignoring llm_out message type={llm_msg.get('type')} trace={llm_msg.get('trace_id')}")
                except Exception:
                    continue

def start_mcp(embedding_model="all-MiniLM-L6-v2", K_RETRIEVE=50, K_RERANK=10, llm_model="llama3.2:1b"):
    b = MCPBroker(embedding_model=embedding_model, K_RETRIEVE=K_RETRIEVE, K_RERANK=K_RERANK, llm_model=llm_model)
    b.start()
    return b, b.get_queues_for_coordinator()