"""
SPLADE-RAG: Sparse Lexical and Expansion Model for Retrieval-Augmented Generation

A production-ready RAG system using SPLADE sparse retrieval.
"""

__version__ = "1.0.0"
__author__ = "RAG Project"

from .encoder import SPLADEEncoder
from .index import SPLADEIndex, ElasticsearchSPLADEIndex
from .retriever import SPLADERetriever, HybridRetriever
from .rag import SPLADERAG
from .utils import chunk_text, analyze_sparse_vector, preprocess_for_splade

__all__ = [
    "SPLADEEncoder",
    "SPLADEIndex",
    "ElasticsearchSPLADEIndex",
    "SPLADERetriever",
    "HybridRetriever",
    "SPLADERAG",
    "chunk_text",
    "analyze_sparse_vector",
    "preprocess_for_splade",
]