# Auto-safety-pro
# 🚗 Auto Safety Pro

**Hybrid Search and Retrieval-Augmented Generation over NHTSA Vehicle Safety Data**

M.Sc. Data Science thesis, University of Europe for Applied Sciences, 2026.
📄 [Read the paper](docs/Large_Language_Model_Search_Over_Vehicle_Recalls_and_Complaints.pdf) · 🌐 [Live demo](TODO-add-hugging-face-spaces-link)

<!-- TODO: add a screenshot or GIF of the dashboard here -->
<!-- ![Auto Safety Pro dashboard](docs/screenshots/dashboard.png) -->

---

## The Problem

Public vehicle safety databases such as NHTSA's hold hundreds of thousands of recalls, consumer complaints and defect investigations, but they are searched with keyword-only tools. The same failure gets described in many ways: "engine stall", "loss of propulsion" and "engine shutdown while driving" can all refer to one defect. A keyword search for one phrase misses the others, so analysts end up reading through records by hand.

## The Solution

Auto Safety Pro combines **lexical** and **semantic** retrieval, then adds a **RAG** layer that turns the top results into a short, evidence-grounded summary. A user can ask a natural-language question such as *"What are the common issues for the 2016 Tesla Model S?"* and get relevant records, a defect summary and the raw evidence behind it.

## Architecture

```
User query (natural language)
        │
        ▼
Pre-processing ── cleaning, normalisation, field concatenation
        │
   ┌────┴────────────────┐
   ▼                     ▼
Whoosh (BM25)        FAISS index
lexical scores       all-MiniLM-L6-v2 embeddings
   └────┬────────────────┘
        ▼
Weighted score fusion
S = α · S_semantic + (1 − α) · S_lexical,  α ≈ 0.6–0.7
        │
        ▼
Top-k records (k = 10)
        │
        ▼
Flan-T5 RAG summarisation ── evidence-grounded defect summary
        │
        ▼
Interactive dashboard ── summary · failure distribution · raw evidence
```

## Data

| Dataset | Source | Approx. Records |
|---|---|---|
| Recalls | NHTSA Recall Database | 25,000 |
| Complaints | NHTSA Consumer Complaints | 380,000 |
| Investigations | NHTSA ODI Investigations | 12,000 |

All sources are flattened into a common schema: `(ID, make, model, year, component, summary, source)`.

Raw data is **not committed** to this repository. See [`data/README.md`](data/README.md) for download instructions from NHTSA's public datasets.

## Key Design Decisions

- **Hybrid over single-method retrieval.** Keyword search is precise but misses reworded descriptions. Semantic search improves recall but can drift to loosely related records. Fusing both kept exact matches while adding contextually similar ones.
- **all-MiniLM-L6-v2 for embeddings.** A good trade-off between semantic quality and inference speed on a single T4 GPU.
- **Fusion weight α ≈ 0.6–0.7**, tuned empirically on this corpus.
- **RAG constrained to retrieved evidence.** Flan-T5 is prompted to summarise only the top-k records, which reduces hallucination and keeps every summary traceable to source documents. Critical in a safety domain.
- **Offline index build.** Embeddings and indexes are built once (40–50 min on Colab T4) and persisted to disk, so retrieval and summarisation run in seconds.

## Results

Evaluated with Precision@10, Recall@10, MRR and nDCG@10 on manually written natural-language and keyword queries.

<!-- TODO: fill in with numbers from your evaluation notebook -->
| System | Precision@10 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| BM25 (keyword-only baseline) | TODO | TODO | TODO | TODO |
| Semantic only (FAISS) | TODO | TODO | TODO | TODO |
| **Hybrid (ours)** | **TODO** | **TODO** | **TODO** | **TODO** |

**Findings:** the hybrid system performed best overall, keeping lexical precision while expanding recall, with the largest gain in MRR, meaning relevant records were ranked higher.

## Project Structure

```
auto-safety-pro/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   └── README.md              # how to download NHTSA data
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_build_indexes.ipynb
│   ├── 03_hybrid_retrieval_evaluation.ipynb
│   └── 04_rag_summarisation.ipynb
├── src/
│   ├── preprocess.py
│   ├── lexical_index.py
│   ├── semantic_index.py
│   ├── hybrid_search.py
│   └── rag.py
├── app/
│   └── app.py                 # dashboard
└── docs/
    ├── paper.pdf
    └── screenshots/
```

## Getting Started

```bash
git clone https://github.com/Thilakraju/auto-safety-pro.git
cd auto-safety-pro
pip install -r requirements.txt

# 1. Download NHTSA data into data/raw/ (see data/README.md)
# 2. Run notebooks 01 → 04 in order, or:
python app/app.py
```

## Tech Stack

Python · Pandas · NumPy · Sentence-Transformers · FAISS · Whoosh (BM25) · Hugging Face Transformers (Flan-T5) · Google Colab (T4 GPU)

## Limitations and Future Work

- English-language records only; multilingual embeddings would enable cross-country comparison.
- Static offline index; an incremental or streaming index would let new complaints arrive without a full rebuild.
- Fixed weighted fusion; learning-to-rank could replace the hand-tuned α.
- Domain fine-tuning of the summarisation model on regulatory documents.

## Author

**Thilak Raju** · [LinkedIn](https://linkedin.com/in/thilak-raju) · thilakraju97@gmail.com
