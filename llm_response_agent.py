import requests
from datetime import datetime
import json
import time

def _log(msg):
    print(f"[{datetime.now().isoformat()}] [LLMResponseAgent] {msg}")

class LLMResponseAgent:
    def __init__(self, in_q, out_q, model_name="llama3.2:1b", base_url="http://localhost:11434"):
        self.in_q = in_q
        self.out_q = out_q
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        _log(f"Initialized LLMResponseAgent model={self.model_name} base_url={self.base_url}")

    def _extract_from_json_obj(self, j):
        if not isinstance(j, dict):
            return None
        for k in ("response", "result", "output", "text"):
            if k in j and isinstance(j[k], (str, int, float)):
                return str(j[k])
        if "choices" in j and isinstance(j["choices"], list):
            parts = []
            for c in j["choices"]:
                if isinstance(c, dict):
                    if "text" in c and isinstance(c["text"], str):
                        parts.append(c["text"])
                    elif "message" in c:
                        parts.append(str(c["message"]))
                    elif "content" in c:
                        parts.append(str(c["content"]))
                else:
                    parts.append(str(c))
            if parts:
                return "\n".join(parts)
        if "result" in j and isinstance(j["result"], dict):
            for k in ("content", "text", "message", "response"):
                if k in j["result"]:
                    return str(j["result"][k])
        return None

    def call_ollama(self, prompt: str, max_tokens=512, temperature=0.0, stream_timeout=180):
        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model_name, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
        headers = {"Content-Type": "application/json"}
        _log(f"Calling Ollama: POST {url} (prompt len={len(prompt)} chars)")
        t0 = time.time()
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=stream_timeout, stream=True)
        except Exception as e:
            _log(f"HTTP request failed: {e}")
            return f"[Error calling Ollama: {e}]"
        t1 = time.time()
        _log(f"Ollama HTTP status={resp.status_code} (setup {t1-t0:.2f}s)")

        collected = []
        raw_lines = []
        try:
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                raw_lines.append(line)
                line = line.strip()
                parsed = None
                try:
                    parsed = json.loads(line)
                except Exception:
                    if "}{ " in line or "}{" in line:
                        parts = []
                        pieces = []
                        buf = ""
                        for ch in line:
                            buf += ch
                        pieces = line.replace('}{', '}\n{').splitlines()
                        for p in pieces:
                            try:
                                obj = json.loads(p)
                                parts.append(obj)
                            except Exception:
                                continue
                        if parts:
                            for obj in parts:
                                s = self._extract_from_json_obj(obj)
                                if s:
                                    collected.append(s)
                    else:
                        collected.append(line)
                else:
                    if isinstance(parsed, list):
                        for item in parsed:
                            s = self._extract_from_json_obj(item)
                            if s:
                                collected.append(s)
                    else:
                        s = self._extract_from_json_obj(parsed)
                        if s:
                            collected.append(s)
            if not collected:
                try:
                    resp_text = resp.text
                    j = json.loads(resp_text)
                    s = self._extract_from_json_obj(j)
                    if s:
                        collected.append(s)
                    else:
                        if isinstance(j, dict) and "choices" in j:
                            for c in j["choices"]:
                                if isinstance(c, dict) and "text" in c:
                                    collected.append(c["text"])
                except Exception:
                    txt = resp.text or ""
                    if txt:
                        collected.append(txt)
        except Exception as e:
            _log(f"Error while streaming/parsing Ollama response: {e}")
            try:
                txt = resp.text or ""
                if txt:
                    collected.append(txt)
            except Exception:
                collected.append(f"[error reading response: {e}]")

        # join pieces to form final answer
        answer = "".join([c for c in collected if c is not None])
        if not answer:
            try:
                answer = (resp.text or "")[:4000]
            except Exception:
                answer = "[No answer returned]"
        _log(f"call_ollama produced answer length={len(answer)} (collected parts={len(collected)})")
        preview = answer[:300].replace("\n", " ")
        _log(f"Ollama preview: {preview}")
        return answer

    def run_once(self, msg):
        if msg.get("type") == "RETRIEVAL_RESULT":
            trace = msg.get("trace_id")
            query = msg["payload"]["query"]
            retrieved = msg["payload"].get("retrieved_context", [])
            _log(f"Received RETRIEVAL_RESULT trace={trace} | {len(retrieved)} chunks")

            ctx_preview = "; ".join([f"{c['meta'].get('source')}#{c['meta'].get('chunk_index')}" for c in retrieved[:5]])
            _log(f"Context preview: {ctx_preview}")

            prompt_parts = [
                "You are a helpful assistant. Use ONLY the provided context to answer the question.",
                f"QUESTION: {query}",
                "CONTEXT:",
            ]
            for i, c in enumerate(retrieved, start=1):
                src = c["meta"].get("source", "unknown")
                text = c.get("text", "")
                prompt_parts.append(f"[{src}]:\n{text}")
            prompt = "\n\n".join(prompt_parts)

            answer = self.call_ollama(prompt)
            if answer is None or (isinstance(answer, str) and not answer.strip()):
                _log("Ollama returned empty/whitespace; replacing with placeholder message.")
                answer = "[Ollama returned no usable answer.]"

            answer_str = str(answer)
            preview = answer_str[:200].replace("\n", " ")
            _log(f"Ollama returned {len(answer_str)} chars. Preview: {preview}")

            resp = {
                "type": "LLM_ANSWER",
                "sender": "LLMResponseAgent",
                
                "trace_id": trace,
                "payload": {"answer": answer, "retrieved_context": retrieved, "query": query},
            }
            self.out_q.put(resp)