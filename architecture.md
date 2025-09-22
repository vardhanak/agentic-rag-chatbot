

# MCP Agent Swimlane Diagram (horizontal timing)

Below is a Mermaid **sequence diagram** (swimlane-style) that shows the interactions and approximate timing between UI, MCP Broker, IngestionAgent, RetrievalAgent, and LLMResponseAgent / Ollama based on your logs. Paste this block into a Mermaid viewer (mermaid.live or VS Code Mermaid plugin) to render.

```mermaid
sequenceDiagram
  participant UI as UI
  participant Broker as MCP Broker
  participant Ingest as IngestionAgent
  participant Retrieval as RetrievalAgent
  participant LLM as LLMResponseAgent
  participant Ollama as Ollama

  note over UI,Broker: 2025-09-22T14:00:25 — Auto-start MCP broker and agents
  UI->>Broker: auto-start MCP broker & agents
  Broker-->>Ingest: start IngestionAgent (pid=6644)
  Broker-->>Retrieval: start RetrievalAgent (pid=7076)
  Broker-->>LLM: start LLMResponseAgent (pid=12268)
  Broker-->>Broker: enter broker loop (routing messages)
  UI->>Broker: MCP started & queues wired to session_state.queues

  Note over LLM: 2025-09-22T14:00:26 — LLM initialized (llama3.2:1b)
  LLM-->>LLM: Initialized model=llama3.2:1b

  Note over Retrieval: embedding & reranker load (dim=384)
  Retrieval-->>Retrieval: load embedding model (dim=384)
  Retrieval-->>Retrieval: load reranker model (cross-encoder/...)

  %% Later auto-start (duplicate run)
  note over UI,Broker: 2025-09-22T14:00:36 — second auto-start sequence
  UI->>Broker: auto-start MCP broker & agents (repeat)
  Broker-->>Ingest: start IngestionAgent (pid=11132)
  Broker-->>Retrieval: start RetrievalAgent (pid=20168)
  Broker-->>LLM: start LLMResponseAgent (pid=18064)
  Broker-->>Broker: enter broker loop
  UI->>Broker: MCP started & queues wired

  Retrieval-->>Retrieval: embedding model loaded (dim=384)
  Retrieval-->>Retrieval: reranker loaded

  %% Ingestion flow
  UI->>Broker: Ingest pressed (2025-09-22T14:01:01) — upload 2 files
  Broker->>Ingest: POST UPLOAD_DOCS (trace=upload-1758529861320-a8cc96) files=2
  Ingest-->>Ingest: Parsed 44 chunks (AMZN PDF)
  Ingest-->>Ingest: Parsed 46 chunks (MICROSOFT transcript)
  Ingest-->>Broker: INGESTION_COMPLETE (90 chunks)
  Broker->>Retrieval: Auto-forward 90 chunks (trace=...)
  Retrieval-->>Retrieval: CHUNKS_ADD -> start indexing
  Retrieval-->>Retrieval: Indexed 90 chunks (total vectors now: 90)
  Retrieval-->>Retrieval: Indexed another 90 chunks (total vectors now: 180)

  %% Query / retrieval / LLM
  UI->>Broker: User prompt (2025-09-22T14:01:50) — query about Amazon Q2 2025
  Broker->>Retrieval: RETRIEVAL_REQUEST (search top 50)
  Retrieval-->>Retrieval: FAISS search -> 50 candidates
  Retrieval->>Retrieval: Reranking top 10 with CrossEncoder
  Retrieval-->>LLM: RETRIEVAL_RESULT (10 chunks)
  LLM-->>Ollama: POST /api/generate (prompt len=8095)
  Ollama-->>LLM: HTTP 200 (setup 59.14s) — reply collected parts=134
  LLM-->>Broker: LLM_ANSWER (preview length=418 chars)
  Broker-->>UI: Return LLM_ANSWER

  Note over LLM,UI: End of recorded interaction (2025-09-22T14:03:00)
```


