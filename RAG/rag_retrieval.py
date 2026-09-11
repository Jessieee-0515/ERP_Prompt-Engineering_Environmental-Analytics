import os
import json
import hashlib
from pathlib import Path

import numpy as np
from openai import OpenAI


# Configuration
EMBED_MODEL = "text-embedding-3-small"
EXTENDED_TOP_K = 4                        # Number of supplementary chunks retrieved from the extended layer
KB_PATH = Path(__file__).parent / "knowledge_base.json"
CACHE_PATH = Path(__file__).parent / "embeddings_cache.json"

TASK_QUERY = (
    "Generate a Python implementation of the Air2Water lake surface "
    "temperature ODE model: governing equation, parameters, numerical "
    "solver, calibration against observed data, goodness-of-fit, "
    "visualization."
)

client = OpenAI()
# 1. Load the knowledge base
def load_knowledge_base() -> list[dict]:
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# 2. Embedding + local cache

def _text_hash(text: str) -> str:
    """Compute a short hash for the text, used as the cache key; the cache
    automatically becomes invalid if the content changes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def get_embedding(text: str, cache: dict) -> list[float]:
    """
    Return the embedding vector for the text, preferring the cache; the API is
    only called if the cache misses.
    """
    key = _text_hash(text)
    if key in cache:
        return cache[key]

    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    vector = response.data[0].embedding

    cache[key] = vector
    return vector


def embed_all_chunks(chunks: list[dict]) -> dict:
    """
    Compute the embedding for every chunk in the knowledge base, returning
    {chunk_id: vector}. Automatically uses/updates the local cache file.
    """
    cache = _load_cache()
    chunk_vectors = {}

    for chunk in chunks:
        vector = get_embedding(chunk["text"], cache)
        chunk_vectors[chunk["chunk_id"]] = vector

    _save_cache(cache)
    return chunk_vectors

# 3. Similarity retrieval

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(query: str, chunks: list[dict], chunk_vectors: dict,
             extended_top_k: int = EXTENDED_TOP_K) -> list[dict]:

    cache = _load_cache()
    query_vector = get_embedding(query, cache)
    _save_cache(cache)

    core_chunks = [c for c in chunks if c.get("layer") == "core"]
    extended_chunks = [c for c in chunks if c.get("layer") == "extended"]

    # Core layer: keep all, attach similarity score for record-keeping only
    core_result = []
    for chunk in core_chunks:
        sim = cosine_similarity(query_vector, chunk_vectors[chunk["chunk_id"]])
        core_result.append({**chunk, "similarity": sim, "selected_by": "core_mandatory"})

    # Extended layer: actual retrieval ranking
    extended_scored = []
    for chunk in extended_chunks:
        sim = cosine_similarity(query_vector, chunk_vectors[chunk["chunk_id"]])
        extended_scored.append({**chunk, "similarity": sim, "selected_by": "retrieved"})
    extended_scored.sort(key=lambda c: c["similarity"], reverse=True)
    extended_result = extended_scored[:extended_top_k]

    return core_result + extended_result

# 4. Assemble the final P6 Prompt

TASK_INSTRUCTION = """You are provided with two paired daily time series: Ta (air temperature, °C) and Tw
(water temperature, °C), each as a 1D numpy array of equal length N, assumed already
loaded as Ta_obs and Tw_obs. Using the Air2Water model structure described in the
reference material above, generate a complete Python implementation of the model.

Your implementation should:
(1) define the governing ODE following the equations given above;
(2) implement all model parameters as described;
(3) numerically solve the ODE;
(4) calibrate the model parameters against the provided Tw_obs data using any
    optimization method of your choice;
(5) report a goodness-of-fit metric;
(6) produce a visualization comparing simulated vs observed water temperature.

Return:
1. Python code
2. A brief explanation of the implementation and your choice of calibration method."""


def build_p6_prompt(retrieved_chunks: list[dict]) -> tuple[str, list[str]]:

    reference_block = "\n\n".join(
        f"[{c['source']} — {c['title']}]\n{c['text']}"
        for c in retrieved_chunks
    )

    prompt = (
        "The following is reference material about the Air2Water lake surface "
        "water temperature model, automatically retrieved based on the task "
        "requirements:\n\n"
        f"{reference_block}\n\n---\n\n{TASK_INSTRUCTION}"
    )

    used_ids = [c["chunk_id"] for c in retrieved_chunks]
    return prompt, used_ids

# 5. Main function for other scripts to call

def get_p6_rag_prompt(extended_top_k: int = EXTENDED_TOP_K) -> tuple[str, list[str]]:
    """
    All-in-one function: load knowledge base -> embedding -> hybrid retrieval
    (core layer mandatorily kept + extended layer retrieval ranking) -> assemble
    prompt.
    batch_generate_prompts.py can call this function directly to get the P6
    prompt text.
    """
    chunks = load_knowledge_base()
    chunk_vectors = embed_all_chunks(chunks)
    retrieved = retrieve(TASK_QUERY, chunks, chunk_vectors,
                          extended_top_k=extended_top_k)
    return build_p6_prompt(retrieved)

# Run directly from the command line: inspect the retrieval results

if __name__ == "__main__":
    chunks = load_knowledge_base()
    print(f"Knowledge base has {len(chunks)} chunks in total, computing embeddings...")

    chunk_vectors = embed_all_chunks(chunks)
    print("Embedding complete (cached to embeddings_cache.json)\n")

    retrieved = retrieve(TASK_QUERY, chunks, chunk_vectors,
                          extended_top_k=EXTENDED_TOP_K)

    core_used = [c for c in retrieved if c["selected_by"] == "core_mandatory"]
    ext_used = [c for c in retrieved if c["selected_by"] == "retrieved"]

    print(f"Core layer (mandatorily kept, {len(core_used)} in total):")
    for c in core_used:
        print(f"  {c['chunk_id']} — {c['title']} (reference similarity: {c['similarity']:.4f})")

    print(f"\nExtended layer (embedding retrieval, top-{EXTENDED_TOP_K}):")
    for c in ext_used:
        print(f"  [{c['similarity']:.4f}] {c['chunk_id']} — {c['title']} "
              f"({c['source']})")

    prompt, used_ids = build_p6_prompt(retrieved)

    print("\n" + "=" * 60)
    print("Assembled final P6 Prompt:")
    print("=" * 60)
    print(prompt)

    print("\n" + "=" * 60)
    print(f"chunk_ids used (for logging): {used_ids}")