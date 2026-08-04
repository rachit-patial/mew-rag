
---

# Mew-Rag

A lightweight, memory-safe, and high-performance **Retrieval-Augmented Generation (RAG)** engine built with Rust.

This system combines local embedded vector processing via **FastEmbed (ONNX)**, scalable vector storage using **Qdrant**, and cloud-hosted LLM execution (via **Groq / OpenAI API**) to answer contextual queries without consuming massive local CPU/GPU resources.

---

## 🏗️ Architecture Overview

```text
       [ Raw Document ]
              │
              ▼
    1. Text Chunking (text-splitter)
              │
              ▼
    2. Local Embeddings (FastEmbed / all-MiniLM-L6-v2-Q)
              │
              ▼
    3. Vector Indexing ────► [ Qdrant DB (gRPC :6334) ]
                                    ▲
                                    │
    4. User Query ──────────────────┘
              │ (Top-K Semantic Search)
              ▼
    5. Context-Augmented Prompt
              │
              ▼
    6. LLM Generation ─────► [ Groq / Cloud API ] ──► Final Response

```

---

## 🚀 Key Features

* **Zero Python Overhead:** Native Rust performance using `tokio` async runtime.
* **Local Ingestion:** Chunking and vector generation happen completely on-device using quantized ONNX embeddings (`384-dimensional`).
* **Persistent Vector Storage:** Powered by Qdrant running via Docker with persistent volume storage.
* **Cloud-Delegated LLM Execution:** Prevents high CPU/RAM system strain by using Groq's high-speed inference API (`llama-3.3-70b`).
* **Environment Configuration:** Secure secret management using `dotenvy`.

---

## 📂 Directory Layout

```text
rag_ingestion/
├── .env                  # API keys and environment variables
├── docker-compose.yml    # Qdrant service configuration
├── Cargo.toml            # Dependencies and features
└── src/
    ├── main.rs           # Pipeline orchestrator
    ├── models.rs         # Common schemas (RawDocument, Chunk)
    ├── ingestion.rs      # Parsing, chunking, fastembed, & Qdrant upserts
    └── retrieval.rs      # Query vectorization, top-k search & LLM generation

```

---

## 🛠️ Prerequisites

* **Rust:** (Edition 2021) installed via [`rustup`](https://rustup.rs/)
* **Docker & Docker Compose:** For running the Qdrant database instance
* **Groq API Key:** Free tier key from [console.groq.com](https://console.groq.com) (or an OpenAI API key)

---

## ⚙️ Setup & Configuration

### 1. Clone the repository & set up environment

```bash
git clone https://github.com/your-username/rust-rag-pipeline.git
cd rust-rag-pipeline

```

### 2. Configure `.env`

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here

```

### 3. Start Qdrant Vector Store

Launch Qdrant using Docker Compose:

```bash
docker compose up -d

```

> 💡 *You can verify Qdrant is running by accessing the Web UI at [`http://localhost:6333/dashboard`](http://localhost:6333/dashboard).*

---

## 🚦 Running the Application

Execute the pipeline using Cargo:

```bash
cargo run

```

### What happens on execution:

1. **Ingestion Phase:** Reads raw text, chunks it using `text-splitter`, generates vectors via local `fastembed` ONNX execution, and upserts points to Qdrant over gRPC (`:6334`).
2. **Retrieval Phase:** Converts user text query to an embedding vector, performs a cosine similarity search against Qdrant to pull top-K matches, and constructs an augmented prompt.
3. **Generation Phase:** Sends the prompt to Groq API (`llama-3.3-70b-versatile`) and prints the final contextual answer.

---

## 📦 Core Dependencies

| Crate | Purpose |
| --- | --- |
| `tokio` | Async runtime for non-blocking execution |
| `qdrant-client` | Official Rust SDK for Qdrant Vector DB |
| `fastembed` | Local embedding generation via ONNX runtime |
| `text-splitter` | Token & character-bounded text chunking |
| `serde` / `serde_json` | Data serialization & payload parsing |
| `reqwest` | HTTP client for cloud LLM API requests |
| `dotenvy` | Environment variable loader |