"""
SPLADE Retriever Module

Implements retrieval logic including hybrid retrieval (SPLADE + dense)
and re-ranking capabilities.
"""

import heapq
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union, Callable
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SPLADERetriever:
    """
    Core SPLADE retriever using inverted index.

    Combines SPLADE encoder with SPLADEIndex for end-to-end retrieval.
    """

    def __init__(
        self,
        index: "SPLADEIndex",
        encoder: "SPLADEEncoder",
        tokenizer=None
    ):
        self.index = index
        self.encoder = encoder
        self.tokenizer = tokenizer or encoder.tokenizer

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        return_metadata: bool = True
    ) -> List[Dict]:
        """
        Retrieve top-k documents for a query.

        Args:
            query: Query string
            top_k: Number of results
            min_score: Minimum score threshold
            return_metadata: Include document metadata in results

        Returns:
            List of result dictionaries with doc_id, score, content, metadata
        """
        # Encode query
        query_vec = self.encoder.encode_queries([query])[0]

        # Retrieve from index
        results = self.index.retrieve(query_vec, top_k=top_k, min_score=min_score)

        # Format results
        formatted = []
        for doc_id, score in results:
            result = {"doc_id": doc_id, "score": float(score)}
            if return_metadata:
                meta = self.index.get_document(doc_id)
                if meta:
                    result["content"] = meta.get("_content", "")
                    result["metadata"] = {k: v for k, v in meta.items() if k != "_content"}
            formatted.append(result)

        return formatted

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
        min_score: float = 0.0
    ) -> List[List[Dict]]:
        """Retrieve for multiple queries."""
        query_vecs = self.encoder.encode_queries(queries)
        all_results = []

        for query_vec in query_vecs:
            results = self.index.retrieve(query_vec, top_k=top_k, min_score=min_score)
            formatted = []
            for doc_id, score in results:
                meta = self.index.get_document(doc_id)
                formatted.append({
                    "doc_id": doc_id,
                    "score": float(score),
                    "content": meta.get("_content", "") if meta else "",
                    "metadata": {k: v for k, v in meta.items() if k != "_content"} if meta else {}
                })
            all_results.append(formatted)

        return all_results

    def retrieve_with_vectors(
        self,
        query: str,
        top_k: int = 10
    ) -> Tuple[List[Dict], Dict[int, float]]:
        """Retrieve and also return the query sparse vector."""
        query_vec = self.encoder.encode_queries([query])[0]
        results = self.retrieve(query, top_k=top_k)
        return results, query_vec


class DenseRetriever:
    """
    Interface for dense retrievers (DPR, ANCE, Contriever, etc.)

    Implement this interface for your dense retrieval backend.
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._load_model()

    def _load_model(self):
        """Load dense encoder model."""
        raise NotImplementedError

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode queries to dense vectors."""
        raise NotImplementedError

    def encode_documents(self, docs: List[str]) -> np.ndarray:
        """Encode documents to dense vectors."""
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve top-k documents."""
        raise NotImplementedError

    def add_documents(self, doc_ids: List[str], vectors: np.ndarray, metadata: List[Dict]):
        """Add documents to index."""
        raise NotImplementedError


class HybridRetriever:
    """
    Hybrid retriever combining SPLADE (sparse) and dense retrieval.

    Supports multiple fusion strategies:
    - RRF (Reciprocal Rank Fusion)
    - Weighted score combination
    - Linear combination
    - Concat + re-rank
    """

    def __init__(
        self,
        splade_retriever: SPLADERetriever,
        dense_retriever: Optional[DenseRetriever] = None,
        fusion: str = "rrf",
        alpha: float = 0.5,
        rrf_k: int = 60,
        top_k_candidates: int = 100
    ):
        """
        Initialize hybrid retriever.

        Args:
            splade_retriever: SPLADE retriever instance
            dense_retriever: Dense retriever instance (optional)
            fusion: Fusion method ('rrf', 'weighted', 'linear', 'concat')
            alpha: Weight for dense scores (1-alpha for sparse) in weighted/linear
            rrf_k: RRF parameter
            top_k_candidates: Number of candidates from each retriever before fusion
        """
        self.splade = splade_retriever
        self.dense = dense_retriever
        self.fusion = fusion
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.top_k_candidates = top_k_candidates

        if dense_retriever is None and fusion != "splade_only":
            logger.warning("No dense retriever provided, falling back to SPLADE only")
            self.fusion = "splade_only"

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Hybrid retrieval with score fusion.

        Returns:
            List of fused results with combined scores
        """
        if self.fusion == "splade_only" or self.dense is None:
            return self.splade.retrieve(query, top_k=top_k, min_score=min_score)

        # Get candidates from both retrievers
        splade_results = self.splade.retrieve(
            query, top_k=self.top_k_candidates, min_score=min_score
        )

        dense_results = self.dense.retrieve(query, top_k=self.top_k_candidates)

        # Convert to dict for fusion
        splade_dict = {r["doc_id"]: r for r in splade_results}
        dense_dict = {doc_id: {"doc_id": doc_id, "score": score} for doc_id, score in dense_results}

        # Apply fusion
        if self.fusion == "rrf":
            fused = self._rrf_fusion(splade_results, dense_results, top_k)
        elif self.fusion == "weighted":
            fused = self._weighted_fusion(splade_dict, dense_dict, top_k)
        elif self.fusion == "linear":
            fused = self._linear_fusion(splade_dict, dense_dict, top_k)
        elif self.fusion == "concat":
            fused = self._concat_fusion(splade_results, dense_results, top_k)
        else:
            raise ValueError(f"Unknown fusion: {self.fusion}")

        return fused

    def _rrf_fusion(
        self,
        splade_results: List[Dict],
        dense_results: List[Tuple[str, float]],
        top_k: int
    ) -> List[Dict]:
        """Reciprocal Rank Fusion."""
        scores = defaultdict(float)
        doc_info = {}

        # SPLADE ranks
        for rank, result in enumerate(splade_results):
            doc_id = result["doc_id"]
            scores[doc_id] += 1.0 / (self.rrf_k + rank + 1)
            doc_info[doc_id] = result

        # Dense ranks
        for rank, (doc_id, score) in enumerate(dense_results):
            scores[doc_id] += 1.0 / (self.rrf_k + rank + 1)
            if doc_id not in doc_info:
                doc_info[doc_id] = {"doc_id": doc_id, "score": score, "content": "", "metadata": {}}

        # Top-k
        top_docs = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])

        results = []
        for doc_id, fused_score in top_docs:
            result = doc_info[doc_id].copy()
            result["fused_score"] = fused_score
            result["retrieval_method"] = "hybrid_rrf"
            results.append(result)

        return results

    def _weighted_fusion(
        self,
        splade_dict: Dict[str, Dict],
        dense_dict: Dict[str, Dict],
        top_k: int
    ) -> List[Dict]:
        """Weighted score combination with normalization."""
        all_doc_ids = set(splade_dict.keys()) | set(dense_dict.keys())

        # Normalize scores to [0, 1] per retriever
        splade_scores = {k: v["score"] for k, v in splade_dict.items()}
        dense_scores = {k: v["score"] for k, v in dense_dict.items()}

        splade_norm = self._minmax_normalize(splade_scores)
        dense_norm = self._minmax_normalize(dense_scores)

        fused_scores = {}
        for doc_id in all_doc_ids:
            s_score = splade_norm.get(doc_id, 0)
            d_score = dense_norm.get(doc_id, 0)
            fused_scores[doc_id] = (1 - self.alpha) * s_score + self.alpha * d_score

        top_docs = heapq.nlargest(top_k, fused_scores.items(), key=lambda x: x[1])

        results = []
        for doc_id, fused_score in top_docs:
            base = splade_dict.get(doc_id) or dense_dict.get(doc_id)
            result = base.copy() if base else {"doc_id": doc_id, "content": "", "metadata": {}}
            result["fused_score"] = fused_score
            result["splade_score"] = splade_dict.get(doc_id, {}).get("score", 0)
            result["dense_score"] = dense_dict.get(doc_id, {}).get("score", 0)
            result["retrieval_method"] = "hybrid_weighted"
            results.append(result)

        return results

    def _linear_fusion(
        self,
        splade_dict: Dict[str, Dict],
        dense_dict: Dict[str, Dict],
        top_k: int
    ) -> List[Dict]:
        """Linear combination without normalization (raw scores)."""
        all_doc_ids = set(splade_dict.keys()) | set(dense_dict.keys())

        fused_scores = {}
        for doc_id in all_doc_ids:
            s_score = splade_dict.get(doc_id, {}).get("score", 0)
            d_score = dense_dict.get(doc_id, {}).get("score", 0)
            fused_scores[doc_id] = (1 - self.alpha) * s_score + self.alpha * d_score

        top_docs = heapq.nlargest(top_k, fused_scores.items(), key=lambda x: x[1])

        results = []
        for doc_id, fused_score in top_docs:
            base = splade_dict.get(doc_id) or dense_dict.get(doc_id)
            result = base.copy() if base else {"doc_id": doc_id, "content": "", "metadata": {}}
            result["fused_score"] = fused_score
            result["retrieval_method"] = "hybrid_linear"
            results.append(result)

        return results

    def _concat_fusion(
        self,
        splade_results: List[Dict],
        dense_results: List[Tuple[str, float]],
        top_k: int
    ) -> List[Dict]:
        """Concatenate results and re-rank (requires cross-encoder)."""
        # Simple concatenation - deduplicate by doc_id
        seen = set()
        combined = []

        for result in splade_results:
            if result["doc_id"] not in seen:
                result["source"] = "splade"
                combined.append(result)
                seen.add(result["doc_id"])

        for doc_id, score in dense_results:
            if doc_id not in seen:
                combined.append({
                    "doc_id": doc_id,
                    "score": score,
                    "content": "",
                    "metadata": {},
                    "source": "dense"
                })
                seen.add(doc_id)

        # Sort by score
        combined.sort(key=lambda x: x["score"], reverse=True)

        for r in combined[:top_k]:
            r["retrieval_method"] = "hybrid_concat"

        return combined[:top_k]

    @staticmethod
    def _minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
        """Min-max normalization to [0, 1]."""
        if not scores:
            return {}
        values = list(scores.values())
        min_v, max_v = min(values), max(values)
        if max_v == min_v:
            return {k: 1.0 for k in scores}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}


class ReRanker:
    """
    Cross-encoder re-ranker for improving retrieval quality.

    Uses a cross-encoder model to re-score query-document pairs.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 512
    ):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name, device=device, max_length=max_length)
            self.batch_size = batch_size
            self.max_length = max_length
        except ImportError:
            raise ImportError("sentence-transformers required: pip install sentence-transformers")

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Re-rank documents using cross-encoder.

        Args:
            query: Query string
            documents: List of document dicts with 'content' field
            top_k: Return top-k after re-ranking

        Returns:
            Re-ranked documents with 'rerank_score' field
        """
        if not documents:
            return []

        # Prepare pairs
        pairs = [(query, doc.get("content", "")) for doc in documents]

        # Score in batches
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        # Add scores and sort
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        documents.sort(key=lambda x: x["rerank_score"], reverse=True)

        if top_k:
            documents = documents[:top_k]

        return documents


class RetrievalPipeline:
    """
    Complete retrieval pipeline with optional re-ranking.

    Combines retriever + re-ranker for production quality.
    """

    def __init__(
        self,
        retriever: Union[SPLADERetriever, HybridRetriever],
        reranker: Optional[ReRanker] = None,
        retriever_top_k: int = 100,
        rerank_top_k: int = 10
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.retriever_top_k = retriever_top_k
        self.rerank_top_k = rerank_top_k

    def search(self, query: str, final_top_k: int = 10) -> List[Dict]:
        """
        Full search pipeline: retrieve -> re-rank -> return top-k.

        Args:
            query: Query string
            final_top_k: Final number of results

        Returns:
            Ranked documents
        """
        # First-stage retrieval
        candidates = self.retriever.retrieve(
            query,
            top_k=self.retriever_top_k
        )

        # Re-rank if available
        if self.reranker and candidates:
            candidates = self.reranker.rerank(query, candidates, top_k=final_top_k)
        else:
            candidates = candidates[:final_top_k]

        return candidates

    def batch_search(
        self,
        queries: List[str],
        final_top_k: int = 10
    ) -> List[List[Dict]]:
        """Batch search for multiple queries."""
        return [self.search(q, final_top_k) for q in queries]