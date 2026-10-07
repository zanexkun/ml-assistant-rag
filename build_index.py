"""
Reads the downloaded articles, splits them into chunks, embeds them,
and saves the index to disk. Run this once after fetch_wiki.py.
"""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

DOCS_DIR = "docs"
MODEL_NAME = "all-MiniLM-L6-v2"
MIN_CHUNK_CHARS = 100


def load_chunks(docs_dir):
    """Split every article into paragraphs, keeping track of the source."""
    chunks, sources = [], []

    for fname in os.listdir(docs_dir):
        path = os.path.join(docs_dir, fname)
        with open(path, encoding="utf-8") as f:
            text = f.read()

        parts = text.split("\n\n")
        chunks.extend(parts)
        sources.extend([fname] * len(parts))

    return chunks, sources


def filter_chunks(chunks, sources, min_chars):
    """Drop headings, separators and formula fragments, which are all short."""
    kept_chunks, kept_sources = [], []

    for i, chunk in enumerate(chunks):
        if len(chunk) < min_chars:
            continue
        kept_chunks.append(chunk)
        kept_sources.append(sources[i])

    return kept_chunks, kept_sources


def main():
    chunks, sources = load_chunks(DOCS_DIR)
    print("raw chunks:", len(chunks))

    chunks, sources = filter_chunks(chunks, sources, MIN_CHUNK_CHARS)
    print("kept chunks:", len(chunks))

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(chunks, normalize_embeddings=True,
                              show_progress_bar=True)
    print("embeddings:", embeddings.shape)

    np.save("embeddings.npy", embeddings)
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "sources": sources}, f)

    print("saved embeddings.npy and chunks.json")


if __name__ == "__main__":
    main()
