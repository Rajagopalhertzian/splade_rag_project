"""
Utility functions for SPLADE-RAG.
"""

import re
from typing import List, Dict, Optional, Any
from transformers import BertTokenizer


def preprocess_for_splade(text: str, max_length: int = 256) -> str:
    """
    Preprocess text for optimal SPLADE encoding.

    SPLADE works best with:
    - Natural language sentences
    - Proper punctuation
    - Lowercased (BERT is uncased)
    - No excessive special characters
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove excessive punctuation but keep sentence structure
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\']', ' ', text)

    # Fix multiple punctuation
    text = re.sub(r'([\.\!\?])\1+', r'\1', text)

    # Strip and ensure proper ending
    text = text.strip()
    if text and text[-1] not in '.!?':
        text += '.'

    return text


def chunk_text(
    text: str,
    chunk_size: int = 200,
    overlap: int = 50,
    tokenizer_name: str = "bert-base-uncased",
    min_chunk_size: int = 50
) -> List[str]:
    """
    Chunk text into overlapping segments optimized for SPLADE.

    Args:
        text: Input text
        chunk_size: Target tokens per chunk
        overlap: Token overlap between chunks
        tokenizer_name: Tokenizer to use
        min_chunk_size: Minimum tokens per chunk

    Returns:
        List of text chunks
    """
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    stride = chunk_size - overlap

    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i:i + chunk_size]

        if len(chunk_tokens) < min_chunk_size:
            # Merge with previous chunk if too small
            if chunks:
                # Extend previous chunk
                prev_tokens = tokenizer.encode(chunks[-1], add_special_tokens=False)
                combined = prev_tokens + chunk_tokens
                if len(combined) <= chunk_size * 1.5:  # Allow some overflow
                    chunks[-1] = tokenizer.decode(combined, skip_special_tokens=True)
                    continue

        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunk_text = chunk_text.strip()

        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def chunk_by_sentences(
    text: str,
    max_tokens: int = 200,
    tokenizer_name: str = "bert-base-uncased"
) -> List[str]:
    """
    Chunk text by sentence boundaries for semantic coherence.
    """
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    # Split by sentences (simple heuristic)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = len(tokenizer.encode(sent, add_special_tokens=False))

        if current_tokens + sent_tokens > max_tokens and current_chunk:
            # Finalize current chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_tokens = sent_tokens
        else:
            current_chunk.append(sent)
            current_tokens += sent_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def analyze_sparse_vector(
    sparse_vec: Dict[int, float],
    tokenizer: BertTokenizer,
    top_n: int = 20
) -> Dict[str, Any]:
    """
    Analyze a SPLADE sparse vector.

    Returns:
        Dictionary with statistics and top terms
    """
    if not sparse_vec:
        return {
            "num_terms": 0,
            "total_weight": 0.0,
            "top_terms": [],
            "weight_stats": {"max": 0, "mean": 0, "min": 0}
        }

    sorted_terms = sorted(sparse_vec.items(), key=lambda x: x[1], reverse=True)

    return {
        "num_terms": len(sparse_vec),
        "total_weight": sum(sparse_vec.values()),
        "top_terms": [
            {
                "token": tokenizer.decode([tid]),
                "token_id": tid,
                "weight": weight
            }
            for tid, weight in sorted_terms[:top_n]
        ],
        "weight_stats": {
            "max": max(sparse_vec.values()),
            "mean": sum(sparse_vec.values()) / len(sparse_vec),
            "min": min(sparse_vec.values())
        }
    }


def compare_vectors(
    vec1: Dict[int, float],
    vec2: Dict[int, float],
    tokenizer: BertTokenizer,
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Compare two sparse vectors (e.g., query vs document).
    """
    # Common terms
    common_terms = set(vec1.keys()) & set(vec2.keys())
    vec1_only = set(vec1.keys()) - set(vec2.keys())
    vec2_only = set(vec2.keys()) - set(vec1.keys())

    def format_terms(term_ids):
        return sorted(
            [(tokenizer.decode([tid]), vec1.get(tid, 0) + vec2.get(tid, 0)) for tid in term_ids],
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

    return {
        "common_terms": format_terms(common_terms),
        "query_only": format_terms(vec1_only),
        "doc_only": format_terms(vec2_only),
        "overlap_ratio": len(common_terms) / max(len(vec1), len(vec2), 1),
        "dot_product": sum(vec1.get(t, 0) * vec2.get(t, 0) for t in common_terms)
    }


def expand_query_terms(
    query: str,
    encoder: "SPLADEEncoder",
    tokenizer: BertTokenizer,
    top_k: int = 20
) -> Dict[str, Any]:
    """
    Show how SPLADE expands a query with additional terms.
    """
    # Get original tokens
    original_tokens = tokenizer.encode(query, add_special_tokens=False)
    original_set = set(original_tokens)

    # Get SPLADE vector
    sparse_vec = encoder.encode_queries([query])[0]

    # Separate original vs expanded
    original = {}
    expanded = {}

    for tid, weight in sparse_vec.items():
        if tid in original_set and tid not in [0, 101, 102, 103]:  # Special tokens
            original[tid] = weight
        elif tid not in [0, 101, 102, 103]:
            expanded[tid] = weight

    return {
        "original_query": query,
        "original_terms": [
            {"token": tokenizer.decode([tid]), "weight": w}
            for tid, w in sorted(original.items(), key=lambda x: x[1], reverse=True)[:top_k]
        ],
        "expanded_terms": [
            {"token": tokenizer.decode([tid]), "weight": w}
            for tid, w in sorted(expanded.items(), key=lambda x: x[1], reverse=True)[:top_k]
        ],
        "expansion_ratio": len(expanded) / max(len(original), 1)
    }


def format_retrieval_results(
    results: List[Dict],
    max_content_length: int = 200
) -> str:
    """Format retrieval results for display."""
    lines = []
    for i, result in enumerate(results):
        doc_id = result.get("doc_id", "unknown")
        score = result.get("score", result.get("fused_score", result.get("rerank_score", 0)))
        content = result.get("content", result.get("_content", ""))
        content_preview = content[:max_content_length] + ("..." if len(content) > max_content_length else "")

        lines.append(f"  {i+1}. [{score:.4f}] {doc_id}")
        lines.append(f"     {content_preview}")
        lines.append("")

    return "\n".join(lines)


def load_documents_from_directory(
    directory: str,
    extensions: List[str] = None,
    recursive: bool = True
) -> List[Dict]:
    """
    Load documents from a directory.

    Args:
        directory: Path to directory
        extensions: File extensions to load (default: .txt, .md, .json)
        recursive: Search subdirectories

    Returns:
        List of document dictionaries
    """
    import os
    from pathlib import Path

    extensions = extensions or ['.txt', '.md', '.json', '.pdf']

    documents = []
    path = Path(directory)

    if recursive:
        files = path.rglob("*")
    else:
        files = path.glob("*")

    for file_path in files:
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            try:
                if file_path.suffix.lower() == '.json':
                    import json
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        for i, item in enumerate(data):
                            if isinstance(item, dict) and 'content' in item:
                                documents.append({
                                    "id": f"{file_path.stem}_{i}",
                                    "content": item['content'],
                                    "metadata": {**item.get('metadata', {}), "source_file": str(file_path)}
                                })
                    elif isinstance(data, dict) and 'content' in data:
                        documents.append({
                            "id": file_path.stem,
                            "content": data['content'],
                            "metadata": {**data.get('metadata', {}), "source_file": str(file_path)}
                        })
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    documents.append({
                        "id": file_path.stem,
                        "content": content,
                        "metadata": {"source_file": str(file_path)}
                    })
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    return documents