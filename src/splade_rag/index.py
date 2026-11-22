"""
SPLADE Index Module

Implements inverted index for sparse vector retrieval
and Elasticsearch integration for production deployment.
"""

import json
import pickle
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import heapq
import logging

logger = logging.getLogger(__name__)


class SPLADEIndex:
    """
    In-memory inverted index for SPLADE sparse vectors.

    Structure: token_id -> List[(doc_id, weight)]
    Supports fast retrieval via sparse dot product.
    """

    def __init__(self):
        # Inverted index: token_id -> List[(doc_id, weight)]
        self.index: Dict[int, List[Tuple[str, float]]] = defaultdict(list)

        # Document metadata storage
        self.doc_metadata: Dict[str, Dict] = {}

        # Document vectors (for re-scoring)
        self.doc_vectors: Dict[str, Dict[int, float]] = {}

        # Statistics
        self.stats = {
            "num_docs": 0,
            "num_terms": 0,
            "total_postings": 0,
            "avg_terms_per_doc": 0.0
        }

    def add_documents(
        self,
        doc_ids: List[str],
        sparse_vecs: List[Dict[int, float]],
        metadata: Optional[List[Dict]] = None,
        contents: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to the index.

        Args:
            doc_ids: Unique document identifiers
            sparse_vecs: List of {token_id: weight} dictionaries
            metadata: Optional metadata for each document
            contents: Optional document contents for storage
        """
        metadata = metadata or [{}] * len(doc_ids)
        contents = contents or [""] * len(doc_ids)

        for doc_id, vec, meta, content in zip(doc_ids, sparse_vecs, metadata, contents):
            # Store metadata and content
            self.doc_metadata[doc_id] = meta
            self.doc_metadata[doc_id]["_content"] = content
            self.doc_vectors[doc_id] = vec

            # Add to inverted index
            for token_id, weight in vec.items():
                self.index[token_id].append((doc_id, weight))

        self._update_stats()

    def _update_stats(self):
        """Update index statistics."""
        self.stats["num_docs"] = len(self.doc_metadata)
        self.stats["num_terms"] = len(self.index)
        self.stats["total_postings"] = sum(len(postings) for postings in self.index.values())
        self.stats["avg_terms_per_doc"] = (
            self.stats["total_postings"] / self.stats["num_docs"]
            if self.stats["num_docs"] > 0 else 0
        )

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the index (lazy deletion)."""
        if doc_id not in self.doc_metadata:
            return False

        # Mark as deleted
        self.doc_metadata[doc_id]["_deleted"] = True
        return True

    def compact(self) -> int:
        """
        Rebuild index removing deleted documents.

        Returns:
            Number of documents removed
        """
        deleted_ids = {
            doc_id for doc_id, meta in self.doc_metadata.items()
            if meta.get("_deleted", False)
        }

        if not deleted_ids:
            return 0

        # Rebuild index
        new_index = defaultdict(list)
        for token_id, postings in self.index.items():
            for doc_id, weight in postings:
                if doc_id not in deleted_ids:
                    new_index[token_id].append((doc_id, weight))

        # Clean metadata and vectors
        for doc_id in deleted_ids:
            self.doc_metadata.pop(doc_id, None)
            self.doc_vectors.pop(doc_id, None)

        self.index = new_index
        self._update_stats()

        return len(deleted_ids)

    def get_candidates(
        self,
        query_vec: Dict[int, float],
        min_score: float = 0.0
    ) -> Dict[str, float]:
        """
        Get candidate documents with scores via inverted index lookup.

        Args:
            query_vec: Query sparse vector {token_id: weight}
            min_score: Minimum score threshold

        Returns:
            Dictionary of doc_id -> score
        """
        scores = defaultdict(float)

        for token_id, q_weight in query_vec.items():
            if token_id in self.index:
                for doc_id, d_weight in self.index[token_id]:
                    # Skip deleted documents
                    if self.doc_metadata.get(doc_id, {}).get("_deleted", False):
                        continue
                    scores[doc_id] += q_weight * d_weight

        # Filter by minimum score
        if min_score > 0:
            scores = {k: v for k, v in scores.items() if v >= min_score}

        return dict(scores)

    def retrieve(
        self,
        query_vec: Dict[int, float],
        top_k: int = 10,
        min_score: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k documents for a query vector.

        Args:
            query_vec: Query sparse vector
            top_k: Number of documents to retrieve
            min_score: Minimum score threshold

        Returns:
            List of (doc_id, score) tuples sorted by score descending
        """
        scores = self.get_candidates(query_vec, min_score)

        if not scores:
            return []

        # Use heapq for efficient top-k
        top_docs = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
        return top_docs

    def batch_retrieve(
        self,
        query_vecs: List[Dict[int, float]],
        top_k: int = 10
    ) -> List[List[Tuple[str, float]]]:
        """Retrieve for multiple queries."""
        return [self.retrieve(qv, top_k) for qv in query_vecs]

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document metadata and content."""
        meta = self.doc_metadata.get(doc_id)
        if meta and meta.get("_deleted", False):
            return None
        return meta

    def get_document_vector(self, doc_id: str) -> Optional[Dict[int, float]]:
        """Get document sparse vector."""
        if self.doc_metadata.get(doc_id, {}).get("_deleted", False):
            return None
        return self.doc_vectors.get(doc_id)

    def save(self, path: Union[str, Path], format: str = "json") -> None:
        """
        Save index to disk.

        Args:
            path: Directory path
            format: 'json' or 'pickle'
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if format == "json":
            # Save inverted index
            index_data = {
                str(k): v for k, v in self.index.items()
            }
            with open(path / "index.json", "w") as f:
                json.dump(index_data, f)

            # Save metadata
            with open(path / "metadata.json", "w") as f:
                json.dump(self.doc_metadata, f)

            # Save vectors (optional, for re-scoring)
            vec_data = {k: v for k, v in self.doc_vectors.items()}
            with open(path / "vectors.json", "w") as f:
                json.dump(vec_data, f)

        elif format == "pickle":
            with open(path / "index.pkl", "wb") as f:
                pickle.dump(self.index, f)
            with open(path / "metadata.pkl", "wb") as f:
                pickle.dump(self.doc_metadata, f)
            with open(path / "vectors.pkl", "wb") as f:
                pickle.dump(self.doc_vectors, f)

        # Save stats
        with open(path / "stats.json", "w") as f:
            json.dump(self.stats, f)

        logger.info(f"Index saved to {path} ({format})")

    @classmethod
    def load(cls, path: Union[str, Path], format: str = "json") -> "SPLADEIndex":
        """Load index from disk."""
        path = Path(path)
        index = cls()

        if format == "json":
            with open(path / "index.json", "r") as f:
                index.index = {int(k): v for k, v in json.load(f).items()}

            with open(path / "metadata.json", "r") as f:
                index.doc_metadata = json.load(f)

            vec_path = path / "vectors.json"
            if vec_path.exists():
                with open(vec_path, "r") as f:
                    index.doc_vectors = {k: v for k, v in json.load(f).items()}

        elif format == "pickle":
            with open(path / "index.pkl", "rb") as f:
                index.index = pickle.load(f)
            with open(path / "metadata.pkl", "rb") as f:
                index.doc_metadata = pickle.load(f)
            with open(path / "vectors.pkl", "rb") as f:
                index.doc_vectors = pickle.load(f)

        stats_path = path / "stats.json"
        if stats_path.exists():
            with open(stats_path, "r") as f:
                index.stats = json.load(f)
        else:
            index._update_stats()

        logger.info(f"Index loaded from {path} ({format})")
        return index

    def __len__(self) -> int:
        return self.stats["num_docs"]

    def __repr__(self) -> str:
        return (f"SPLADEIndex(docs={self.stats['num_docs']}, "
                f"terms={self.stats['num_terms']}, "
                f"postings={self.stats['total_postings']})")


class ElasticsearchSPLADEIndex:
    """
    Elasticsearch-backed SPLADE index for production deployment.

    Uses rank_features field type for efficient sparse vector search.
    """

    def __init__(
        self,
        es_url: str = "http://localhost:9200",
        index_name: str = "splade_rag",
        timeout: int = 30
    ):
        try:
            from elasticsearch import Elasticsearch
            self.es = Elasticsearch([es_url], request_timeout=timeout)
            self.index_name = index_name
            self._create_index()
        except ImportError:
            raise ImportError("elasticsearch package required: pip install elasticsearch")

    def _create_index(self) -> None:
        """Create Elasticsearch index with proper mappings."""
        if not self.es.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "splade_vector": {"type": "rank_features"},
                        "metadata": {"type": "object", "enabled": True}
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "index.codec": "best_compression"
                }
            }
            self.es.indices.create(index=self.index_name, body=mapping)
            logger.info(f"Created Elasticsearch index: {self.index_name}")

    def add_documents(self, documents: List[Dict]) -> None:
        """
        Add documents to Elasticsearch.

        Args:
            documents: List of dicts with keys:
                - id: document ID
                - content: text content
                - splade_vec: {token_id: weight} sparse vector
                - metadata: optional metadata dict
        """
        from elasticsearch.helpers import bulk

        actions = []
        for doc in documents:
            # Convert sparse dict to rank_features format
            rank_features = {
                str(token_id): weight
                for token_id, weight in doc["splade_vec"].items()
            }

            actions.append({
                "_index": self.index_name,
                "_id": doc["id"],
                "_source": {
                    "content": doc["content"],
                    "splade_vector": rank_features,
                    "metadata": doc.get("metadata", {})
                }
            })

        bulk(self.es, actions)
        self.es.indices.refresh(index=self.index_name)
        logger.info(f"Indexed {len(documents)} documents")

    def search(
        self,
        query_vec: Dict[int, float],
        top_k: int = 10,
        min_score: float = 0.0,
        filter_query: Optional[Dict] = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search using SPLADE sparse vector.

        Args:
            query_vec: Query sparse vector
            top_k: Number of results
            min_score: Minimum score threshold
            filter_query: Optional Elasticsearch filter query

        Returns:
            List of (doc_id, score, metadata) tuples
        """
        # Build rank_feature queries for each token
        should_clauses = []
        for token_id, weight in query_vec.items():
            should_clauses.append({
                "rank_feature": {
                    "field": f"splade_vector.{token_id}",
                    "saturation": {"pivot": 1.0},
                    "boost": weight
                }
            })

        query = {
            "size": top_k,
            "min_score": min_score,
            "query": {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1
                }
            }
        }

        if filter_query:
            query["query"]["bool"]["filter"] = filter_query

        response = self.es.search(index=self.index_name, body=query)

        results = []
        for hit in response["hits"]["hits"]:
            results.append((
                hit["_id"],
                hit["_score"],
                hit["_source"].get("metadata", {})
            ))

        return results

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document by ID."""
        try:
            response = self.es.get(index=self.index_name, id=doc_id)
            return {
                "id": response["_id"],
                "content": response["_source"]["content"],
                "metadata": response["_source"].get("metadata", {})
            }
        except Exception:
            return None

    def delete_document(self, doc_id: str) -> bool:
        """Delete document by ID."""
        try:
            self.es.delete(index=self.index_name, id=doc_id)
            return True
        except Exception:
            return False

    def stats(self) -> Dict:
        """Get index statistics."""
        stats = self.es.indices.stats(index=self.index_name)
        count = self.es.count(index=self.index_name)["count"]
        return {
            "num_docs": count,
            "index_size_bytes": stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"]
        }


def create_index_from_documents(
    documents: List[Dict],
    encoder,
    index_type: str = "memory",
    **index_kwargs
) -> Union[SPLADEIndex, ElasticsearchSPLADEIndex]:
    """
    Factory function to create and populate an index.

    Args:
        documents: List of {'id': str, 'content': str, 'metadata': dict}
        encoder: SPLADEEncoder instance
        index_type: 'memory' or 'elasticsearch'
        **index_kwargs: Additional arguments for index constructor

    Returns:
        Populated index instance
    """
    # Extract contents
    contents = [doc["content"] for doc in documents]
    doc_ids = [doc["id"] for doc in documents]
    metadata = [doc.get("metadata", {}) for doc in documents]

    # Encode documents
    logger.info(f"Encoding {len(documents)} documents...")
    sparse_vecs = encoder.encode_documents(contents)

    if index_type == "memory":
        index = SPLADEIndex(**index_kwargs)
        index.add_documents(doc_ids, sparse_vecs, metadata, contents)
    elif index_type == "elasticsearch":
        index = ElasticsearchSPLADEIndex(**index_kwargs)
        docs_for_es = [
            {"id": did, "content": cont, "splade_vec": vec, "metadata": meta}
            for did, cont, vec, meta in zip(doc_ids, contents, sparse_vecs, metadata)
        ]
        index.add_documents(docs_for_es)
    else:
        raise ValueError(f"Unknown index_type: {index_type}")

    return index