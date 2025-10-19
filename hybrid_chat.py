# Path: hybrid_chat.py
"""
Hybrid chat CLI (updated with hybrid ranking + provenance).
- Adds hybrid_rank() which combines Pinecone similarity score and graph centrality (degree)
  to re-rank semantic matches before building the prompt.
- Prints clear provenance for each selected source: (name, id, origin, score, degree, hybrid_score)
- Keeps Gemini auto-selection and OpenAI fallback logic from previous version.
- Put your secrets in config.py (see config.py.sample). Do NOT commit keys.
"""

from typing import List, Dict, Any
import sys
import traceback

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from neo4j import GraphDatabase

# Optional: OpenAI fallback
try:
    import openai
except Exception:
    openai = None

# Gemini (google.generativeai) SDK
try:
    import google.generativeai as genai
except Exception:
    genai = None

import config

# -----------------------------
# Config (from config.py)
# -----------------------------
EMBED_MODEL_NAME = getattr(config, "EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
CHAT_MODEL_NAME = getattr(config, "CHAT_MODEL_NAME", "models/gemini-2.5-pro")
TOP_K = getattr(config, "TOP_K", 5)
INDEX_NAME = config.PINECONE_INDEX_NAME
# Weight for mix: alpha * pinecone_score + (1-alpha) * graph_degree_score
HYBRID_ALPHA = getattr(config, "HYBRID_ALPHA", 0.7)

# -----------------------------
# Initialize embedding model
# -----------------------------
try:
    embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
except Exception as e:
    print("ERROR: failed to load embedding model:", e)
    raise

# -----------------------------
# Init Pinecone
# -----------------------------
if not getattr(config, "PINECONE_API_KEY", None):
    print("ERROR: PINECONE_API_KEY missing in config.py")
    sys.exit(1)

pc = Pinecone(api_key=config.PINECONE_API_KEY)
try:
    index = pc.Index(INDEX_NAME)
except Exception as e:
    print("ERROR: Could not connect to Pinecone index. Check INDEX_NAME and PINECONE_API_KEY.")
    print("Exception:", e)
    raise

# -----------------------------
# Init Neo4j
# -----------------------------
try:
    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
except Exception as e:
    print("ERROR: Failed to connect to Neo4j. Check NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in config.py.")
    raise

# -----------------------------
# Init Gemini (if available)
# -----------------------------
available_model_names = []
if genai is not None:
    genai.configure(api_key=getattr(config, "GEMINI_API_KEY", None))
    try:
        models = genai.list_models()
        for m in models:
            if isinstance(m, dict) and "name" in m:
                available_model_names.append(m["name"])
            else:
                available_model_names.append(getattr(m, "name", str(m)))
    except Exception:
        available_model_names = []
else:
    print("NOTE: google.generativeai (Gemini SDK) not installed or unavailable.")

# Auto-select a good Gemini model if configured one is not available
if available_model_names:
    preferred_candidates = [
        "models/gemini-2.5-pro",
        "models/gemini-2.5-pro-preview-06-05",
        "models/gemini-2.5-pro-preview-05-06",
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
    ]
    if CHAT_MODEL_NAME in available_model_names:
        print(f"Using configured CHAT_MODEL_NAME = '{CHAT_MODEL_NAME}'")
    else:
        picked = None
        for cand in preferred_candidates:
            if cand in available_model_names:
                picked = cand
                break
        if not picked:
            for name in available_model_names:
                if "embedding" in name:
                    continue
                picked = name
                break
        if picked:
            print(f"Configured CHAT_MODEL_NAME '{CHAT_MODEL_NAME}' not found. Auto-selecting '{picked}' for this run.")
            CHAT_MODEL_NAME = picked
        else:
            print("No suitable Gemini chat model found in list. Will attempt runtime calls and/or fallback to OpenAI if configured.")

# Model wrapper (may or may not be supported)
chat_model = None
if genai is not None and CHAT_MODEL_NAME:
    try:
        chat_model = genai.GenerativeModel(CHAT_MODEL_NAME)
    except Exception:
        chat_model = None

# -----------------------------
# Optional OpenAI fallback init
# -----------------------------
OPENAI_AVAILABLE = False
if getattr(config, "OPENAI_API_KEY", None) and openai is not None:
    try:
        openai.api_key = config.OPENAI_API_KEY
        OPENAI_AVAILABLE = True
    except Exception:
        OPENAI_AVAILABLE = False

# -----------------------------
# Helper functions
# -----------------------------
def embed_text(text: str) -> List[float]:
    vec = embedding_model.encode(text, show_progress_bar=False)
    return (vec.tolist() if hasattr(vec, "tolist") else list(vec))

def pinecone_query(query_text: str, top_k=TOP_K):
    vec = embed_text(query_text)
    try:
        res = index.query(
            vector=vec,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
    except Exception as e:
        print("ERROR: Pinecone query failed:", e)
        raise

    matches = []
    raw_matches = None
    if isinstance(res, dict):
        raw_matches = res.get("matches") or res.get("results") or []
    else:
        raw_matches = getattr(res, "matches", None) or getattr(res, "results", None) or []

    if raw_matches is None:
        print("WARNING: Pinecone returned unexpected format:", type(res))
        return []

    for m in raw_matches:
        if isinstance(m, dict):
            matches.append({
                "id": m.get("id"),
                "score": m.get("score"),
                "metadata": m.get("metadata", {})
            })
        else:
            matches.append({
                "id": getattr(m, "id", None),
                "score": getattr(m, "score", None),
                "metadata": getattr(m, "metadata", {}) or {}
            })

    print(f"DEBUG: Pinecone top {len(matches)} results:")
    for i, mm in enumerate(matches, start=1):
        print(f"  {i}. id={mm['id']} score={mm.get('score')} meta_name={mm['metadata'].get('name')}")
    return matches

def fetch_graph_context(node_ids: List[str]):
    """
    Fetch neighboring nodes and relations for the provided node ids.
    Returns a list of fact dicts with: source, rel, target_id, target_name, target_desc, labels
    """
    facts = []
    if not node_ids:
        return facts
    with driver.session() as session:
        for nid in node_ids:
            try:
                q = (
                    "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
                    "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
                    "m.name AS name, m.description AS description "
                    "LIMIT 50"
                )
                recs = session.run(q, nid=nid)
                for r in recs:
                    facts.append({
                        "source": nid,
                        "rel": r["rel"],
                        "target_id": r["id"],
                        "target_name": r["name"],
                        "target_desc": (r["description"] or "")[:400],
                        "labels": r["labels"]
                    })
            except Exception as e:
                print(f"WARNING: Neo4j fetch failed for node {nid}: {e}")
    print(f"DEBUG: Graph facts found: {len(facts)}")
    return facts

# -----------------------------
# NEW: hybrid ranking utilities
# -----------------------------
def compute_graph_degrees(graph_facts: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Compute a simple degree (centrality) score for node ids based on occurrences
    in graph_facts (counts as source or target). Returns dict {node_id: degree}.
    """
    deg = {}
    for f in graph_facts:
        src = f.get("source")
        tgt = f.get("target_id")
        if src:
            deg[src] = deg.get(src, 0) + 1
        if tgt:
            deg[tgt] = deg.get(tgt, 0) + 1
    return deg

def hybrid_rank(matches: List[Dict[str, Any]], graph_facts: List[Dict[str, Any]], alpha: float = HYBRID_ALPHA) -> List[Dict[str, Any]]:
    """
    Combine Pinecone similarity score and graph degree into a single hybrid score.
    Returns matches sorted by hybrid score descending. Each returned match has
    an added key 'hybrid_score' and 'graph_degree'.
    Formula: hybrid = alpha * normalized_pinecone_score + (1 - alpha) * normalized_degree
    """
    if not matches:
        return []

    # compute degree map from graph facts
    deg = compute_graph_degrees(graph_facts)
    max_deg = max(deg.values()) if deg else 1

    # extract pinecone scores and normalize
    raw_scores = [m.get("score", 0.0) or 0.0 for m in matches]
    max_score = max(raw_scores) if raw_scores and max(raw_scores) > 0 else 1.0

    ranked = []
    for m, raw in zip(matches, raw_scores):
        node_id = m.get("id")
        pscore = raw / max_score if max_score else 0.0
        gscore = (deg.get(node_id, 0) / max_deg) if max_deg else 0.0
        hybrid = float(alpha) * pscore + (1.0 - float(alpha)) * gscore
        # attach debug info
        m2 = dict(m)  # shallow copy
        m2["graph_degree"] = deg.get(node_id, 0)
        m2["normalized_pinecone_score"] = pscore
        m2["normalized_graph_score"] = gscore
        m2["hybrid_score"] = hybrid
        ranked.append(m2)

    ranked.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return ranked

# -----------------------------
# Prompt building & LLM call (kept similar to previous)
# -----------------------------
def build_prompt(user_query, pinecone_matches, graph_facts):
    """Build a chat prompt combining semantic matches and graph facts with human-friendly snippets."""
    system = (
        "You are a helpful travel assistant. Use the provided semantic search results "
        "and graph facts to answer the user's query briefly and concisely. "
        "Prefer human-readable names and short descriptions. If mentioning a node, include the node's name and id."
    )

    # Pinecone context: include name + short snippet if available
    vec_context_lines = []
    for m in pinecone_matches:
        meta = m.get("metadata", {}) or {}
        name = meta.get("name") or m.get("id") or "unknown"
        snippet = meta.get("snippet") or meta.get("short_description") or meta.get("description") or ""
        if snippet:
            snippet = (snippet[:220] + "...") if len(snippet) > 220 else snippet
            vec_context_lines.append(f"- {name} (id: {m['id']}) — {snippet} [score={m.get('score'):.3f} | deg={m.get('graph_degree',0)} | hybrid={m.get('hybrid_score',0):.3f}]")
        else:
            vec_context_lines.append(f"- {name} (id: {m['id']}) [score={m.get('score'):.3f} | deg={m.get('graph_degree',0)} | hybrid={m.get('hybrid_score',0):.3f}]")

    # Graph context: list neighbor facts with description snippets
    graph_context_lines = []
    for f in graph_facts:
        tname = f.get("target_name") or f.get("target_id") or ""
        tdesc = f.get("target_desc") or ""
        if tdesc:
            tdesc = (tdesc[:220] + "...") if len(tdesc) > 220 else tdesc
            graph_context_lines.append(f"- ({f['source']}) -[{f['rel']}]-> ({f['target_id']}) {tname}: {tdesc}")
        else:
            graph_context_lines.append(f"- ({f['source']}) -[{f['rel']}]-> ({f['target_id']}) {tname}")

    prompt = (
        f"{system}\n\n"
        f"User query: {user_query}\n\n"
        "Top semantic matches (hybrid-ranked):\n" + "\n".join(vec_context_lines[:8]) + "\n\n"
        "Graph facts (neighboring relations):\n" + "\n".join(graph_context_lines[:12]) + "\n\n"
        "Using the above, provide a concise answer tailored to the user's request. "
        "If suggesting an itinerary, give a short day-by-day plan (2-4 bullets) and cite the place names with node ids."
    )
    return prompt

def call_gemini(prompt_text: str):
    """Try to call Gemini (genai). Returns string response or raises."""
    if genai is None:
        raise RuntimeError("Gemini SDK not available in environment.")
    if chat_model is not None:
        try:
            resp = chat_model.generate_content(prompt_text)
            if hasattr(resp, "text"):
                return resp.text
            if isinstance(resp, dict) and "text" in resp:
                return resp["text"]
            return str(resp)
        except Exception:
            pass
        try:
            resp = chat_model.generate(prompt_text)
            if hasattr(resp, "text"):
                return resp.text
            return str(resp)
        except Exception as e:
            raise RuntimeError(f"Gemini call via wrapper failed: {e}")
    try:
        resp = genai.generate(model=CHAT_MODEL_NAME, prompt=prompt_text)
        if isinstance(resp, dict):
            text = resp.get("output") or resp.get("content") or resp.get("text")
            if isinstance(text, list):
                return " ".join([t.get("text", "") if isinstance(t, dict) else str(t) for t in text])
            return str(text)
        return str(resp)
    except Exception as e:
        raise RuntimeError(f"Gemini generate failed: {e}")

def call_openai(prompt_text: str):
    """Fallback to OpenAI chat if available. Returns string response or raises."""
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI fallback not available or openai package not installed.")
    try:
        resp = openai.ChatCompletion.create(
            model=getattr(config, "OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a helpful travel assistant."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=800,
            temperature=0.2,
        )
        choices = resp.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content") or choices[0].get("text") or str(choices[0])
        return str(resp)
    except Exception:
        try:
            resp = openai.Completion.create(
                model=getattr(config, "OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                prompt=prompt_text,
                max_tokens=800,
                temperature=0.2,
            )
            if isinstance(resp, dict):
                choices = resp.get("choices", [])
                if choices:
                    return choices[0].get("text", str(choices[0]))
            return str(resp)
        except Exception as e2:
            raise RuntimeError(f"OpenAI call failed: {e2}")

def call_chat(prompt_text: str) -> str:
    """Primary call: try Gemini, on fail try OpenAI (if configured)."""
    if genai is not None and CHAT_MODEL_NAME:
        try:
            return call_gemini(prompt_text)
        except Exception as e:
            print("Gemini call failed:", e)
            try:
                models = genai.list_models()
                names = []
                for m in models:
                    if isinstance(m, dict) and "name" in m:
                        names.append(m["name"])
                    else:
                        names.append(getattr(m, "name", str(m)))
                print("Available Gemini models (sample):", names[:20])
            except Exception as e2:
                print("Failed to list Gemini models:", e2)
    if OPENAI_AVAILABLE:
        try:
            return call_openai(prompt_text)
        except Exception as e:
            print("OpenAI fallback failed:", e)
            traceback.print_exc()
            return "Sorry — both Gemini and OpenAI calls failed. See console logs."
    return "No LLM backend available. Configure GEMINI_API_KEY or OPENAI_API_KEY in config.py."

# -----------------------------
# Interactive CLI (uses hybrid ranking + provenance)
# -----------------------------
def interactive_chat():
    print("Hybrid travel assistant (embedding local model + Pinecone + Neo4j). Type 'exit' to quit.")
    while True:
        try:
            query = input("\nEnter your travel question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not query or query.lower() in ("exit", "quit"):
            break

        # 1) semantic search (fetch more than TOP_K to allow re-ranking)
        try:
            raw_matches = pinecone_query(query, top_k=max(TOP_K * 3, 10))
        except Exception as e:
            print("Failed to query Pinecone:", e)
            continue

        # 2) fetch graph context for all matched ids
        node_ids = [m["id"] for m in raw_matches if m.get("id")]
        graph_facts = fetch_graph_context(node_ids)

        # 3) hybrid rank matches (combine semantic score + graph degree)
        ranked_matches = hybrid_rank(raw_matches, graph_facts, alpha=HYBRID_ALPHA)

        # 4) select top K for prompt
        top_matches = ranked_matches[:TOP_K]

        # 5) build prompt using the top hybrid-ranked matches
        prompt = build_prompt(query, top_matches, graph_facts)

        # 6) call LLM
        answer = call_chat(prompt)

        # 7) print assistant answer
        print("\n=== Assistant Answer ===\n" + answer + "\n=== End ===\n")

        # 8) PRINT PROVENANCE: friendly list including hybrid scores
        print("Provenance (selected sources):")
        for m in top_matches:
            meta = m.get("metadata", {}) or {}
            name = meta.get("name") or m.get("id")
            snippet = meta.get("snippet") or meta.get("short_description") or meta.get("description") or ""
            s_snip = (snippet[:140] + "...") if snippet and len(snippet) > 140 else snippet
            print(f" - {name} (id:{m.get('id')}) | origin: vector | pinecone_score: {m.get('score'):.4f} | degree: {m.get('graph_degree',0)} | hybrid: {m.get('hybrid_score'):.4f}")
            if s_snip:
                print(f"    snippet: {s_snip}")

        # 9) concise graph facts sample for debugging/provenance
        print("\nGraph facts (sample):")
        for f in graph_facts[:8]:
            tname = f.get("target_name") or f.get("target_id")
            tdesc = f.get("target_desc") or ""
            tdesc_short = (tdesc[:140] + "...") if tdesc and len(tdesc) > 140 else tdesc
            print(f"  - {f['source']} -[{f['rel']}]-> {tname} (id:{f['target_id']})", end="")
            if tdesc_short:
                print(f" — {tdesc_short}")
            else:
                print("")

if __name__ == "__main__":
    interactive_chat()
