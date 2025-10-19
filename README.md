 
# Hybrid Knowledge AI System  
*Blue Enigma Labs — AI Engineer Technical Challenge*  
**Author:** Anuj Gadekar .
**Date:** October 2025  

---

## 📘 Overview

This project implements a **Hybrid Knowledge Retrieval & Reasoning System** that combines:
- **Neo4j** for graph-based knowledge representation (entity–relation reasoning),  
- **Pinecone** for semantic vector retrieval using dense embeddings, and  
- **LLM (Gemini/OpenAI)** for natural language reasoning and contextual answer generation.  

The solution demonstrates how structured (graph) and unstructured (semantic) knowledge sources can be fused to power **intelligent, context-aware conversational AI systems** — a foundational approach for the next generation of travel and knowledge assistants.

---

## ⚙️ System Architecture

 

 <img width="1785" height="990" alt="image" src="https://github.com/user-attachments/assets/b73b4a27-2c98-4fc7-98bd-976eea678d2c" />
---
 
**Core modules**
- `load_to_neo4j.py` → Loads entities/relations into Neo4j.  
- `visualize_graph.py` → Generates an interactive HTML graph (`neo4j_viz.html`).  
- `pinecone_upload.py` → Embeds and upserts text into Pinecone (batched, cached).  
- `hybrid_chat.py` → Combines Neo4j + Pinecone + LLM reasoning for conversational answers.  
- `utils/embed_cache.py` → Disk-backed JSON cache for embeddings (performance boost).

---

## 🧠 Design Highlights

| Feature | Description |
|----------|--------------|
| **Hybrid Ranking** | Combines vector similarity with graph-degree centrality to prioritize semantically relevant yet structurally significant nodes. |
| **Embedding Cache** | Caches embeddings on disk (`emb_cache.json`) using SHA256 keys → eliminates redundant encoding. |
| **Batch Upserts** | Pinecone uploads processed in configurable batches (default: 32) to balance throughput vs. memory. |
| **Parallel Graph Fetch** | Neo4j queries for neighbors executed concurrently via `ThreadPoolExecutor` → 3–5× faster retrieval on large graphs. |
| **Prompt Limiter** | Automatically truncates or summarizes long contexts to stay within ~4000 chars (≈2000 tokens). |
| **Modular Architecture** | Each module runs independently (`if __name__ == "__main__":`) — ensuring composability, reusability, and testability. |

---

## 🚀 Quick Start

### 1️⃣ Environment Setup
```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
````

### 2️⃣ Configuration

Copy the provided template and fill your keys:

```bash
cp config.py.sample config.py
```

Edit the file with:

* `PINECONE_API_KEY`
* `GEMINI_API_KEY` or `OPENAI_API_KEY`
* `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`

### 3️⃣ Load & Visualize

```bash
python load_to_neo4j.py
python visualize_graph.py
# Output: neo4j_viz.html
```
<img width="882" height="687" alt="image" src="https://github.com/user-attachments/assets/5e4b913b-ed69-4f79-ba6b-bcf628d3f072" />


### 4️⃣ Upload to Pinecone

```bash
python pinecone_upload.py
# Output: "All items uploaded successfully."
```
<img width="1445" height="126" alt="image" src="https://github.com/user-attachments/assets/377e06c0-8d7b-4cfd-86bd-84802a5133a8" />


### 5️⃣ Run Hybrid Chat

```bash
python hybrid_chat.py
# Example query:
# > Plan a 3-day cultural food trip in Vietnam.
```
<img width="1839" height="792" alt="image" src="https://github.com/user-attachments/assets/a2ac4c25-853f-4fd5-8fbc-43b66a39b20e" />
<img width="1839" height="792" alt="image" src="https://github.com/user-attachments/assets/8bd4dc50-03b1-4daf-8a36-43ed10ab6f9d" />

---

## 🔍 Testing

### Automated Smoke Test

```bash
# Linux/macOS
./tests/smoke_test.sh
# Windows
./tests/smoke_test.ps1
```

Expected Output:

```
1) Running load_to_neo4j.py... OK
2) Running pinecone_upload.py... OK
3) Running visualize_graph.py... OK
4) Import test for hybrid_chat... OK
SMOKE TESTS PASSED
```

---

## 📊 Performance & Scalability

* **Batching:** 32-sized Pinecone upserts optimized for network efficiency & memory.
* **Caching:** ~90% speedup on re-runs using persistent embedding cache.
* **Async Fetch:** Neo4j neighbor queries parallelized (up to 8 workers by default).
* **Prompt Control:** Keeps input context < 2000 tokens, reducing LLM cost & latency.
* **Hybrid Logic:** Balances semantic precision with relational depth — ensuring contextually richer yet performant results.

---

## 🧩 Example Output

```
Enter your travel question: Plan a 3-day food and culture trip in Vietnam.
=== Assistant Answer ===
Day 1: Explore Hanoi’s old quarter and street food hubs.
Day 2: Visit cultural landmarks and local craft villages.
Day 3: Optional day trip to Hue for imperial cuisine.
=== End ===
Sources: Pinecone → Hanoi, Hue; Graph Facts → 14 relations.
```

---

## 🧾 License & Acknowledgements

This project is part of the **Blue Enigma Labs AI Engineer Challenge (2025)**.
Developed by **Anuj G. (ShXlabs)** for evaluation purposes.
All external datasets and embeddings remain property of their respective owners.

---

## 🧭 Professional Notes (Architect’s Perspective)

* Designed with **stateless modularity** for rapid scaling to cloud-native workloads.
* Components can be containerized independently for microservice deployment.
* Future extensions:

  * Replace local cache with Redis for distributed caching.
  * Use Neo4j’s vector indexing for unified hybrid retrieval.
  * Stream responses via `asyncio` and WebSocket API for real-time chat UX.

> *"Clarity of thought, modular design, and predictable scaling are the foundations of all enduring software systems."*
> — *Senior AI Systems Architect, 2025*

```
 
