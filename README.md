# SPLADE-RAG: Sparse Lexical and Expansion Model for Retrieval-Augmented Generation

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-4.35+-yellow.svg)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready **Retrieval-Augmented Generation (RAG)** system powered by **SPLADE (Sparse Lexical and Expansion Model)** — a state-of-the-art sparse neural retrieval model that combines the efficiency of inverted indexes with the semantic understanding of neural models.

## 🎯 Why SPLADE-RAG?

| Feature | Dense Retrieval (DPR/ANCE) | BM25 | **SPLADE-RAG** |
|---------|---------------------------|------|----------------|
| **Semantic Understanding** | ✅ | ❌ | ✅ |
| **Exact Keyword Matching** | ❌ | ✅ | ✅ |
| **Query Expansion** | ❌ | ❌ | ✅ (Learned) |
| **Interpretability** | ❌ | ✅ | ✅ |
| **Index Size (1M docs)** | ~4.5 GB | ~1.2 GB | **~0.9 GB** |
| **Query Latency** | ~50-100ms | ~5ms | **~12ms** |
| **Infrastructure** | GPU + ANN Index | CPU + Inverted Index | **CPU + Inverted Index** |
| **Re-ranking Friendly** | ✅ | ✅ | ✅ |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd C:\Users\91767\Desktop\rag_projects\splade_rag_project

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Basic Usage

```python
from splade_rag import create_rag_system

# Create RAG system (downloads models on first run)
rag = create_rag_system(
    splade_model="naver/splade-max",           # Best quality SPLADE
    llm_model="microsoft/Phi-3-mini-4k-instruct",  # Efficient LLM
    device="cuda"                              # or "cpu"
)

# Add your documents
documents = [
    {"id": "doc1", "content": "Your document text here...", "metadata": {"source": "wiki"}},
    {"id": "doc2", "content": "Another document...", "metadata": {"source": "web"}},
]
rag.add_documents(documents)

# Ask questions
result = rag.query("What is SPLADE and how does it work?")
print(result.answer)
print(f"Sources: {len(result.citations)}")
```

### Command Line Interface

```bash
# Build index from documents
python scripts/build_index.py --data_dir ./data/documents --output ./data/splade_index

# Query the system
python scripts/query_rag.py --index ./data/splade_index

# Evaluate on test set
python scripts/evaluate.py --index ./data/splade_index --test_data ./data/test.jsonl

# Launch web UI
streamlit run scripts/streamlit_app.py
```

## 📁 Project Structure

```
splade_rag_project/
├── config/
│   └── config.yaml              # Configuration file
├── docs/
│   └── splade.doc               # Comprehensive SPLADE documentation
├── notebooks/
│   └── 01_splade_rag_demo.ipynb # Interactive demo notebook
├── scripts/
│   ├── build_index.py           # Build index from documents
│   ├── query_rag.py             # Interactive query CLI
│   ├── train_splade.py          # Fine-tune SPLADE on custom data
│   ├── evaluate.py              # Evaluate retrieval/RAG quality
│   └── streamlit_app.py         # Web UI
├── src/
│   └── splade_rag/
│       ├── __init__.py
│       ├── encoder.py           # SPLADE model & encoder
│       ├── index.py             # Inverted index & Elasticsearch
│       ├── retriever.py         # Retrieval & hybrid fusion
│       ├── rag.py               # End-to-end RAG pipeline
│       └── utils.py             # Chunking, analysis utilities
├── data/                        # Your data goes here
│   ├── documents/               # Source documents
│   └── splade_index/            # Built index
├── requirements.txt
└── README.md
```

## 🔧 Core Components

### 1. SPLADE Encoder (`encoder.py`)
- Loads pre-trained SPLADE models from HuggingFace Hub
- Supports `splade-max`, `splade-distil`, `splade-cocondenser`, `multi-splade`
- Converts text to sparse vectors with configurable top-k sparsity
- Provides analysis utilities for query expansion visualization

### 2. Inverted Index (`index.py`)
- **In-memory index**: Fast, suitable for <1M documents
- **Elasticsearch index**: Production-ready, horizontal scaling
- Efficient sparse dot product via inverted index lookup
- Supports incremental updates and persistence (JSON/Pickle)

### 3. Retrievers (`retriever.py`)
- `SPLADERetriever`: Pure SPLADE retrieval
- `HybridRetriever`: Combines SPLADE + dense (RRF, weighted, linear fusion)
- `ReRanker`: Cross-encoder re-ranking for quality boost
- `RetrievalPipeline`: Composable retrieve → re-rank pipeline

### 4. RAG System (`rag.py`)
- `SPLADERAG`: Complete RAG with retrieval + LLM generation
- Streaming responses for real-time UX
- Citation tracking with source attribution
- Configurable chunking, retrieval, and generation parameters

## 📚 SPLADE Documentation

See [`docs/splade.doc`](docs/splade.doc) for comprehensive coverage of:

- **Theoretical foundations**: From TF-IDF to learned sparse retrieval
- **Architecture details**: BERT + MLM head + aggregation strategies
- **Training objectives**: Contrastive learning, FLOPS regularization, distillation
- **Variants**: SPLADE v1, v2 (max), distil, ensemble, multilingual, domain-adapted
- **Implementation**: Complete PyTorch code with inference pipeline
- **RAG integration**: Elasticsearch, hybrid retrieval, re-ranking
- **Benchmarks**: MS MARCO, BEIR results vs dense/BM25
- **Best practices**: Chunking, hyperparameters, monitoring, debugging
- **Advanced topics**: Distillation, quantization, incremental indexing

## 🎓 Training Custom SPLADE Models

Fine-tune SPLADE on your domain data:

```bash
# Prepare training data (JSONL format)
# Each line: {"query": "...", "positive": "...", "negatives": ["...", "..."]}

python scripts/train_splade.py \
    --train_data ./data/train.jsonl \
    --val_data ./data/val.jsonl \
    --output ./models/splade-finetuned \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 2e-5
```

Training features:
- Contrastive loss with in-batch negatives
- FLOPS sparsity regularization
- Hard negative mining support
- Gradient accumulation for large effective batch sizes
- Linear warmup + cosine decay scheduling
- Checkpointing and best model saving

## 📊 Evaluation

```bash
# Retrieval metrics (Recall@k, MRR@k, nDCG@k)
python scripts/evaluate.py \
    --index ./data/splade_index \
    --test_data ./data/test.jsonl \
    --eval_retrieval

# End-to-end RAG evaluation
python scripts/evaluate.py \
    --index ./data/splade_index \
    --test_data ./data/test.jsonl \
    --eval_rag
```

Test data format (JSONL):
```json
{"query": "What is SPLADE?", "relevant_docs": ["doc_1", "doc_5"]}
{"question": "How does term expansion work?", "answer": "SPLADE uses MLM head...", "sources": ["doc_2"]}
```

## ⚙️ Configuration

All settings in `config/config.yaml`:

```yaml
model:
  splade_model: "naver/splade-max"
  llm_model: "microsoft/Phi-3-mini-4k-instruct"
  device: "cuda"

retrieval:
  top_k_candidates: 100
  final_top_k: 5
  use_hybrid: false
  fusion_method: "rrf"

reranker:
  enabled: true
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

chunking:
  chunk_size: 200
  chunk_overlap: 50

generation:
  max_new_tokens: 512
  temperature: 0.3
  top_p: 0.9
```

## 🐳 Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/

EXPOSE 8501
CMD ["streamlit", "run", "scripts/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📈 Performance Benchmarks

### MS MARCO Passage (Dev)
| Model | MRR@10 | nDCG@10 | Recall@100 | Index Size | Latency |
|-------|--------|---------|------------|------------|---------|
| BM25 | 0.187 | 0.259 | 0.731 | 1.2 GB | 5ms |
| DPR | 0.326 | 0.398 | 0.887 | 4.5 GB | 50ms |
| ColBERTv2 | 0.397 | 0.473 | 0.938 | 12 GB | 150ms |
| **SPLADE v2 (max)** | **0.379** | **0.453** | **0.931** | **0.9 GB** | **12ms** |

### BEIR Zero-Shot (Average nDCG@10)
| Dataset | BM25 | SPLADE v2 | Contriever |
|---------|------|-----------|------------|
| TREC-COVID | 0.656 | **0.742** | 0.689 |
| NFCorpus | 0.325 | **0.372** | 0.341 |
| FiQA | 0.236 | **0.328** | 0.298 |
| **Average** | 0.412 | **0.482** | 0.458 |

## 🛠️ Advanced Usage

### Custom Chunking Strategy
```python
from splade_rag.utils import chunk_text, chunk_by_sentences

# Token-based chunking (default)
chunks = chunk_text(text, chunk_size=200, overlap=50)

# Sentence-aware chunking for better coherence
chunks = chunk_by_sentences(text, max_tokens=200)
```

### Query Analysis
```python
# Understand how SPLADE expands your query
analysis = encoder.analyze_vector(query_vec, top_n=20)
expansion = expand_query_terms(query, encoder, encoder.tokenizer)

print("Original terms:", expansion['original_terms'])
print("Expanded terms:", expansion['expanded_terms'])
```

### Hybrid Retrieval with Dense Models
```python
from sentence_transformers import SentenceTransformer
import numpy as np

class DenseWrapper:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def add_documents(self, doc_ids, texts):
        self.doc_ids = doc_ids
        self.vectors = self.model.encode(texts)
    
    def retrieve(self, query, top_k=10):
        q_vec = self.model.encode([query])[0]
        scores = self.vectors @ q_vec
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_idx]

# Combine with SPLADE
hybrid = HybridRetriever(splade_retriever, DenseWrapper(), fusion="rrf")
results = hybrid.retrieve("your query", top_k=5)
```

### Elasticsearch Production Index
```python
from splade_rag.index import ElasticsearchSPLADEIndex

es_index = ElasticsearchSPLADEIndex(
    es_url="http://elasticsearch:9200",
    index_name="my_splade_index"
)

# Add documents
es_index.add_documents(documents)

# Search
results = es_index.search(query_vec, top_k=10)
```

## 📖 References

1. **SPLADE v1**: Formal et al., *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking*, SIGIR 2021
2. **SPLADE v2**: Formal et al., *SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval*, arXiv:2109.10086
3. **Distillation**: Formal et al., *From Distillation to Hard Negative Sampling*, arXiv:2205.04733
4. **Official Repo**: https://github.com/naver/splade
5. **Models**: https://huggingface.co/models?search=splade

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a PR

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **NAVER LABS** for the original SPLADE models and research
- **Hugging Face** for model hosting and Transformers library
- **Elastic** for Elasticsearch integration patterns
- The open-source IR community for benchmarks and datasets

---
