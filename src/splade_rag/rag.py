"""
SPLADE-RAG: Complete RAG System

Combines SPLADE retrieval with LLM generation for question answering.
"""

import torch
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field

from .encoder import SPLADEEncoder
from .index import SPLADEIndex, ElasticsearchSPLADEIndex, create_index_from_documents
from .retriever import SPLADERetriever, HybridRetriever, RetrievalPipeline, ReRanker
from .utils import chunk_text, preprocess_for_splade

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for SPLADE-RAG system."""

    # Model settings
    splade_model_path: str = "naver/splade-max"
    splade_tokenizer: str = "bert-base-uncased"
    llm_model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    device: str = "cuda"

    # Retrieval settings
    retrieval_top_k: int = 100
    final_top_k: int = 5
    use_hybrid: bool = False
    dense_model: Optional[str] = None
    fusion_method: str = "rrf"

    # Re-ranking
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Chunking
    chunk_size: int = 200
    chunk_overlap: int = 50

    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9

    # Index
    index_type: str = "memory"  # "memory" or "elasticsearch"
    index_path: Optional[str] = None
    es_url: str = "http://localhost:9200"
    es_index_name: str = "splade_rag"

    def __post_init__(self):
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"


@dataclass
class RAGResult:
    """Result from RAG query."""
    answer: str
    citations: List[Dict] = field(default_factory=list)
    retrieved_docs: List[Dict] = field(default_factory=list)
    query: str = ""
    metadata: Dict = field(default_factory=dict)


class SPLADERAG:
    """
    Complete SPLADE-RAG System.

    Combines sparse retrieval with LLM generation for accurate,
    citation-backed question answering.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all components."""
        logger.info("Initializing SPLADE-RAG components...")

        # SPLADE Encoder
        self.encoder = SPLADEEncoder(
            model_path=self.config.splade_model_path,
            tokenizer_name=self.config.splade_tokenizer,
            device=self.config.device
        )

        # Load or create index
        if self.config.index_path and Path(self.config.index_path).exists():
            logger.info(f"Loading index from {self.config.index_path}")
            self.index = SPLADEIndex.load(self.config.index_path)
        else:
            self.index = SPLADEIndex()

        # Retriever
        self.retriever = SPLADERetriever(self.index, self.encoder)

        # Hybrid retriever (optional)
        if self.config.use_hybrid and self.config.dense_model:
            # Would need DenseRetriever implementation
            logger.warning("Hybrid retrieval requested but dense retriever not implemented")
            self.hybrid_retriever = self.retriever
        else:
            self.hybrid_retriever = self.retriever

        # Re-ranker
        if self.config.use_reranker:
            self.reranker = ReRanker(
                model_name=self.config.reranker_model,
                device=self.config.device
            )
        else:
            self.reranker = None

        # Retrieval pipeline
        self.pipeline = RetrievalPipeline(
            retriever=self.hybrid_retriever,
            reranker=self.reranker,
            retriever_top_k=self.config.retrieval_top_k,
            rerank_top_k=self.config.final_top_k
        )

        # LLM
        self._load_llm()

        logger.info("SPLADE-RAG initialized successfully")

    def _load_llm(self):
        """Load LLM for generation."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                self.config.llm_model_name,
                trust_remote_code=True
            )

            self.llm = AutoModelForCausalLM.from_pretrained(
                self.config.llm_model_name,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                device_map="auto" if self.config.device == "cuda" else None,
                trust_remote_code=True
            )

            if self.config.device == "cpu":
                self.llm = self.llm.to("cpu")

            # Set pad token if not present
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise

    def add_documents(
        self,
        documents: List[Dict],
        save_index: bool = True
    ) -> None:
        """
        Add documents to the RAG system.

        Args:
            documents: List of {'id': str, 'content': str, 'metadata': dict}
            save_index: Whether to save index after adding
        """
        logger.info(f"Adding {len(documents)} documents...")

        # Chunk documents
        all_chunks = []
        chunk_metadata = []

        for doc in documents:
            chunks = chunk_text(
                doc["content"],
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap
            )

            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc['id']}_chunk_{i}"
                all_chunks.append(chunk)
                chunk_metadata.append({
                    **doc.get("metadata", {}),
                    "source_doc_id": doc["id"],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "_content": chunk
                })

        # Encode chunks
        logger.info(f"Encoding {len(all_chunks)} chunks...")
        sparse_vecs = self.encoder.encode_documents(all_chunks)

        # Add to index
        chunk_ids = [m["source_doc_id"] + f"_chunk_{m['chunk_index']}" for m in chunk_metadata]
        self.index.add_documents(chunk_ids, sparse_vecs, chunk_metadata, all_chunks)

        # Save index
        if save_index and self.config.index_path:
            self.index.save(self.config.index_path)
            logger.info(f"Index saved to {self.config.index_path}")

    def add_documents_from_files(
        self,
        file_paths: List[Union[str, Path]],
        save_index: bool = True
    ) -> None:
        """Add documents from text/markdown files."""
        documents = []
        for path in file_paths:
            path = Path(path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            documents.append({
                "id": path.stem,
                "content": content,
                "metadata": {"source_file": str(path)}
            })

        self.add_documents(documents, save_index)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_context: bool = False
    ) -> RAGResult:
        """
        Answer a question using RAG.

        Args:
            question: User question
            top_k: Override final_top_k
            return_context: Include full context in result

        Returns:
            RAGResult with answer and citations
        """
        top_k = top_k or self.config.final_top_k

        # Retrieve relevant documents
        logger.info(f"Retrieving for query: {question[:100]}...")
        retrieved = self.pipeline.search(question, final_top_k=top_k)

        if not retrieved:
            return RAGResult(
                answer="I couldn't find any relevant information to answer your question.",
                citations=[],
                retrieved_docs=[],
                query=question
            )

        # Build context with citations
        context_parts = []
        citations = []

        for i, doc in enumerate(retrieved):
            content = doc.get("content", doc.get("_content", ""))
            context_parts.append(f"[Source {i+1}] {content}")

            citations.append({
                "source_id": i + 1,
                "doc_id": doc.get("doc_id", ""),
                "score": doc.get("score", doc.get("fused_score", doc.get("rerank_score", 0))),
                "metadata": {k: v for k, v in doc.items() if k not in ["content", "_content"]}
            })

        context = "\n\n".join(context_parts)

        # Generate answer
        answer = self._generate_answer(question, context)

        return RAGResult(
            answer=answer,
            citations=citations,
            retrieved_docs=retrieved if return_context else [],
            query=question,
            metadata={
                "num_sources": len(citations),
                "retrieval_method": "hybrid" if self.config.use_hybrid else "splade",
                "reranked": self.config.use_reranker
            }
        )

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM."""
        prompt = self._build_prompt(question, context)

        inputs = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.llm.device)

        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.llm_tokenizer.pad_token_id,
                eos_token_id=self.llm_tokenizer.eos_token_id
            )

        # Decode only the generated part
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self.llm_tokenizer.decode(generated, skip_special_tokens=True)

        return answer.strip()

    def _build_prompt(self, question: str, context: str) -> str:
        """Build prompt for LLM."""
        return f"""<|system|>
You are a helpful assistant that answers questions based on the provided context.
Always cite your sources using [Source X] format.
If the context doesn't contain enough information, say so.
</s>
<|user|>
Context:
{context}

Question: {question}

Answer:</s>
<|assistant|>
"""

    def stream_query(self, question: str, top_k: Optional[int] = None):
        """
        Stream answer tokens for real-time display.

        Yields:
            Token strings as they are generated
        """
        top_k = top_k or self.config.final_top_k

        # Retrieve
        retrieved = self.pipeline.search(question, final_top_k=top_k)

        if not retrieved:
            yield "I couldn't find any relevant information to answer your question."
            return

        # Build context
        context_parts = []
        for i, doc in enumerate(retrieved):
            content = doc.get("content", doc.get("_content", ""))
            context_parts.append(f"[Source {i+1}] {content}")
        context = "\n\n".join(context_parts)

        # Stream generation
        prompt = self._build_prompt(question, context)
        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.llm.device)

        from transformers import TextIteratorStreamer
        import threading

        streamer = TextIteratorStreamer(
            self.llm_tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            pad_token_id=self.llm_tokenizer.pad_token_id
        )

        thread = threading.Thread(target=self.llm.generate, kwargs=generation_kwargs)
        thread.start()

        for token in streamer:
            yield token

        thread.join()

    def save_index(self, path: Optional[str] = None) -> None:
        """Save the index to disk."""
        path = path or self.config.index_path
        if path:
            self.index.save(path)
            logger.info(f"Index saved to {path}")

    def get_stats(self) -> Dict:
        """Get system statistics."""
        return {
            "index": self.index.stats,
            "config": {
                "splade_model": self.config.splade_model_path,
                "llm_model": self.config.llm_model_name,
                "chunk_size": self.config.chunk_size,
                "retrieval_top_k": self.config.retrieval_top_k,
                "final_top_k": self.config.final_top_k,
                "use_reranker": self.config.use_reranker,
                "use_hybrid": self.config.use_hybrid
            }
        }


def create_rag_system(
    splade_model: str = "naver/splade-max",
    llm_model: str = "microsoft/Phi-3-mini-4k-instruct",
    index_path: Optional[str] = None,
    device: str = "cuda",
    **kwargs
) -> SPLADERAG:
    """
    Factory function to create a SPLADE-RAG system.

    Args:
        splade_model: SPLADE model name or path
        llm_model: LLM model name
        index_path: Path to load/save index
        device: 'cuda' or 'cpu'
        **kwargs: Additional RAGConfig parameters

    Returns:
        Configured SPLADERAG instance
    """
    config = RAGConfig(
        splade_model_path=splade_model,
        llm_model_name=llm_model,
        index_path=index_path,
        device=device,
        **kwargs
    )
    return SPLADERAG(config)