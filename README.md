# Machine Learning Assistant (RAG)

An assistant that answers machine learning questions from a set of Wikipedia articles, and tells you which article each answer came from. If the answer is not in the documents, it says so instead of making something up.

## What it does

The corpus is 62 Wikipedia articles covering deep learning, classical ML, computer vision, NLP and evaluation metrics. A question goes in, the system finds the paragraphs that actually relate to it, and an LLM writes the answer using only those paragraphs.

The point is that it matches meaning, not keywords. Asking "what is the difference between precision and recall" pulls the right paragraphs even though the word "difference" appears in none of them.

## How it works

1. **Fetch**: download the 62 articles as plain text.
2. **Chunk**: split each article on blank lines, which gives 2,992 paragraphs. Drop anything under 100 characters, since those are headings, separator lines and leftover formula fragments. That leaves 2,736 chunks.
3. **Embed**: turn every chunk into a 384 number vector with `all-MiniLM-L6-v2`, normalized so a dot product gives cosine similarity.
4. **Retrieve**: embed the question with the same model, compare it against all 2,736 chunks in one matrix multiply, take the top 5.
5. **Generate**: send those 5 chunks to `openai/gpt-oss-120b` through the Groq API, with an instruction to use only the given context.

The index is built once and saved to disk, so asking a question takes a couple of seconds instead of re-embedding everything.

## Results

Three cases, all measured on the real corpus.

**A question the documents answer:**

```
python ask.py "what is the difference between precision and recall?"
```

Top match scores 0.821, all five chunks come from the precision and recall article, and the answer uses the dog counting example from the article itself.

**A question the documents do not cover:**

```
python ask.py "how do I deploy a model with FastAPI?"
```

Best match 0.20. Nothing in the corpus is about FastAPI, so the system refuses before calling the LLM at all.

**A question about something that does not exist:**

```
python ask.py "what did the 2023 Chen paper conclude about batch normalization?"
```

There is no such paper. Retrieval returns real batch normalization chunks scoring 0.50 to 0.56, so the threshold does not catch this one. The model answers "I don't know" instead of inventing a conclusion.

That last case is the one worth paying attention to.  There are two separate defences here: a score threshold that catches questions with no relevant documents, and a prompt instruction that catches questions where the documents are on topic but do not hold the answer.

**Score ranges I measured on this corpus:**

| Top score | What it means |
|---|---|
| above 0.6 | the answer is in there |
| 0.4 to 0.6 | right topic, may or may not answer |
| below 0.3 | nothing relevant exists |

These are specific to this model and this corpus, not general numbers.

## Setup

```bash
pip install -r requirements.txt
python fetch_wiki.py      # downloads the articles
python build_index.py     # chunks and embeds, saves the index
```

Put your Groq key in a `.env` file:

```
GROQ_API_KEY=your_key_here
```

Then ask anything:

```bash
python ask.py "why do deep networks suffer from vanishing gradients?"
```

## Limitations

**Formulas are lost.** Wikipedia maths does not survive plain text extraction, so the heavy maths articles leave behind LaTeX fragments and dangling words like "where" and "then". The length filter removes the short ones, but longer fragments get through and occasionally show up in results.

**All five slots can come from one article.** For a narrow question that is correct, but it means related material from other articles never reaches the LLM.

**Chunk boundaries split things apart.** A sentence like "then install what it needs:" ends up separated from the block it was introducing. Overlapping chunks would fix this.

**The threshold is tuned on a handful of questions.** 0.3 works for the cases above but it has not been measured against a proper question set with known correct answers.

**Retrieval matches topic better than specifics.** A question about a number, like "what accuracy did it reach", ranks general paragraphs about the topic above the one that actually holds the figure.

## What I would do next

Overlapping chunks, deduplication, a proper evaluation set with known answers so the threshold can be measured rather than guessed, and a FastAPI endpoint so it runs as a service instead of a script.
