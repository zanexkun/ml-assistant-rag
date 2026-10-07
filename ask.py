"""
Answers a question from the indexed articles.

    python ask.py "what is the difference between precision and recall?"

Needs GROQ_API_KEY set in the environment, or in a .env file.
"""

import os
import sys
import json
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

MODEL_NAME = "all-MiniLM-L6-v2"
LLM_NAME = "openai/gpt-oss-120b"
TOP_K = 5
MIN_SCORE = 0.3          # below this, nothing in the corpus is relevant

PROMPT = """Use only the context below to answer the question.
If the context does not contain the answer, say you don't know.

Context:
{context}

Question: {question}"""


def retrieve(question, model, embeddings, chunks, sources):
    """Return the top matching chunks and the best score."""
    q_vector = model.encode([question], normalize_embeddings=True)[0]
    scores = embeddings @ q_vector
    top = np.argsort(scores)[::-1][:TOP_K]

    hits = [(scores[i], chunks[i], sources[i]) for i in top]
    return hits, scores.max()


def main():
    if len(sys.argv) < 2:
        print('usage: python ask.py "your question"')
        return

    question = sys.argv[1]

    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY is not set")
        return

    embeddings = np.load("embeddings.npy")
    with open("chunks.json", encoding="utf-8") as f:
        data = json.load(f)
    chunks, sources = data["chunks"], data["sources"]

    model = SentenceTransformer(MODEL_NAME)
    hits, best = retrieve(question, model, embeddings, chunks, sources)

    if best < MIN_SCORE:
        print(f"No relevant information found (best match {best:.2f}).")
        return

    context = "\n\n".join(chunk for _, chunk, _ in hits)

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=LLM_NAME,
        messages=[{"role": "user",
                   "content": PROMPT.format(context=context, question=question)}],
    )

    print(response.choices[0].message.content)

    print("\nSources:")
    for score, _, source in hits:
        print(f"  {score:.3f}  {source}")


if __name__ == "__main__":
    main()
