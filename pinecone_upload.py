# Path: pinecone_upload.py
"""
Pinecone uploader with embedding cache and batch upserts.

- Uses SentenceTransformer local model for embeddings.
- Uses utils/embed_cache.py to avoid recomputing embeddings.
- Batch size is configurable (PINECONE_BATCH_SIZE in config.py.sample).
  Reason: batching controls memory footprint and API throttling — set to 32 or 64 depending on memory and throughput.
"""

import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import config
import os

# local embed cache util
from utils.embed_cache import get_cached_embedding

# -----------------------------
# Config
# -----------------------------
DATA_FILE = getattr(config, "DATA_FILE", "vietnam_travel_dataset.json")
BATCH_SIZE = int(getattr(config, "PINECONE_BATCH_SIZE", 32))  # <--- configurable: memory & API limit tradeoff
EMBED_MODEL = getattr(config, "EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

INDEX_NAME = getattr(config, "PINECONE_INDEX_NAME", "hybrid-travel-index")
VECTOR_DIM = int(getattr(config, "PINECONE_VECTOR_DIM", 384))

# -----------------------------
# Initialize clients
# -----------------------------
model = SentenceTransformer(EMBED_MODEL)

if not getattr(config, "PINECONE_API_KEY", None):
    raise SystemExit("ERROR: PINECONE_API_KEY missing in config.py")

pc = Pinecone(api_key=config.PINECONE_API_KEY)

# Create index if not exists (may vary depending on Pinecone env & permissions)
try:
    existing_indexes = pc.list_indexes().names() if hasattr(pc.list_indexes(), "names") else pc.list_indexes()
except Exception:
    # fallback: many SDK versions return different structures
    try:
        existing_indexes = pc.list_indexes()
    except Exception:
        existing_indexes = []

if INDEX_NAME not in existing_indexes:
    print(f"Creating managed index: {INDEX_NAME}")
    try:
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    except Exception as e:
        print("WARNING: create_index failed or not permitted:", e)
else:
    print(f"Index {INDEX_NAME} already exists.")

index = pc.Index(INDEX_NAME)

# -----------------------------
# Helper functions
# -----------------------------
def compute_embedding(texts):
    """Compute embeddings for a list of texts using model.encode and the embed cache."""
    embeddings = []
    for t in texts:
        emb = get_cached_embedding(t, lambda s: model.encode(s, show_progress_bar=False))
        embeddings.append(emb)
    return embeddings

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

# -----------------------------
# Main upload
# -----------------------------
def main():
    if not os.path.exists(DATA_FILE):
        raise SystemExit(f"ERROR: Data file not found: {DATA_FILE}")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    items = []
    for node in nodes:
        semantic_text = node.get("semantic_text") or (node.get("description") or "")[:1000]
        if not semantic_text.strip():
            continue
        meta = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name"),
            "city": node.get("city", node.get("region", "")),
            "tags": node.get("tags", []),
            # include a short snippet if available to feed prompts quickly
            "snippet": (node.get("description") or "")[:360]
        }
        items.append((node["id"], semantic_text, meta))

    print(f"Preparing to upsert {len(items)} items to Pinecone in batches of {BATCH_SIZE}...")

    for batch in tqdm(list(chunked(items, BATCH_SIZE)), desc="Uploading batches"):
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metas = [item[2] for item in batch]

        embeddings = compute_embedding(texts)

        vectors = [
            {"id": _id, "values": emb, "metadata": meta}
            for _id, emb, meta in zip(ids, embeddings, metas)
        ]

        try:
            index.upsert(vectors)
        except Exception as e:
            print("ERROR: Pinecone upsert failed for this batch:", e)
            # continue to next batch (don't halt entire upload)
            continue

    print("All items uploaded successfully.")

if __name__ == "__main__":
    main()
