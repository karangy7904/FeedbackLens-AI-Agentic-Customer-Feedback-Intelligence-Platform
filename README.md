https://rag-customer-feedback-huuqzeappppqkgfuymfrjtyb.streamlit.app/
# Customer Feedback RAG — Extension

This extends the original PDF-RAG tutorial project into a tool actually built for
customer feedback analysis, matching the repo's name.

## What's new vs. the original

| Original | Added here |
|---|---|
| Upload any PDF, chat with it | Upload structured feedback (CSV), each row auto-tagged |
| Plain semantic search | Search filtered by **sentiment** and **topic** metadata |
| No analytics | Sentiment/topic breakdown dashboard (Plotly) |
| Single-purpose Q&A | Business-oriented: "what are customers unhappy about with shipping?" |

## Files

- `sentiment_tagging.py` — tags each feedback row with sentiment (positive/negative)
  and topic (zero-shot classification into business categories: shipping, product
  quality, customer service, pricing/billing, usability).
- `ingest_feedback.py` — runs tagging, embeds text with `bge-small-en`, stores in
  a FAISS index with sentiment/topic as retrievable metadata.
- `system.py` — query layer: filtered RAG retrieval + aggregate analytics function.
- `streamlit_app.py` — dashboard UI: upload feedback, see sentiment/topic charts,
  ask filtered questions, see which feedback the answer was based on.
- `sample_feedback.csv` — 18 rows of synthetic customer feedback to test with.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env   # then edit .env and paste in your real HuggingFace token
streamlit run streamlit_app.py
```

`.env` is listed in `.gitignore` so your token is never committed. `.env.example`
is the safe, commit-able template that shows the variable name without a real value.

Or run the pipeline directly:
```bash
python ingest_feedback.py   # tags + builds the index
python system.py            # example filtered query + analytics
```

## Resume bullets (now accurate to what's actually built)

- Built an end-to-end customer feedback analytics tool combining sentiment
  analysis, zero-shot topic classification, and Retrieval-Augmented Generation
  (RAG), enabling filtered natural-language querying (e.g., "negative feedback
  about shipping") over unstructured customer data.

- Implemented a metadata-aware retrieval pipeline in Python using LangChain and
  FAISS, tagging each feedback entry with sentiment (DistilBERT) and topic
  (BART zero-shot classification) to support scoped semantic search beyond
  plain keyword matching.

- Designed and deployed an interactive Streamlit dashboard visualizing
  sentiment/topic distributions (Plotly) and surfacing LLM-generated answers
  with source-level traceability, containerized for reproducible deployment
  with Docker.

## Honest framing note

This still builds on the open-source RAG tutorial architecture (LangChain +
FAISS + HuggingFace + Streamlit). What makes it yours now is the sentiment/topic
tagging layer, the metadata-filtered retrieval, and the analytics dashboard —
be ready to explain those pieces specifically in an interview, since that's the
part that's actually novel versus the base tutorial.
