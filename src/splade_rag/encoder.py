"""
SPLADE Encoder Module

Implements the SPLADE (Sparse Lexical and Expansion Model) encoder
for converting text to sparse vocabulary-based vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SPLADEModel(nn.Module):
    """
    SPLADE Model Architecture

    Uses a BERT backbone with MLM head to produce sparse vectors.
    Supports both max-pooling (SPLADE-max) and sum-pooling aggregation.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        aggregation: str = "max",
        vocab_size: int = 30522,
        dropout: float = 0.1
    ):
        super().__init__()
        self.model_name = model_name
        self.aggregation = aggregation
        self.vocab_size = vocab_size

        # Load BERT backbone
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # MLM Head for vocabulary projection
        self.mlm_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size)
        )

        # Temperature for logits scaling
        self.register_buffer("temperature", torch.tensor(1.0))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass producing sparse vectors.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len] (optional)

        Returns:
            sparse_vec: [batch_size, vocab_size] - non-negative sparse weights
        """
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        # Token-level representations [batch, seq_len, hidden]
        token_embeds = outputs.last_hidden_state

        # MLM head → vocabulary logits [batch, seq_len, vocab]
        logits = self.mlm_head(token_embeds) / self.temperature

        # ReLU for non-negativity (required for inverted index)
        weights = F.relu(logits)

        # Aggregation over sequence dimension
        if self.aggregation == "max":
            # Max pooling: [batch, vocab] - SPLADE-max style
            sparse_vec, _ = weights.max(dim=1)
        elif self.aggregation == "sum":
            # Sum pooling - preserves term frequency info
            sparse_vec = weights.sum(dim=1)
        elif self.aggregation == "log1p_sum":
            # Log(1+sum) for smoother gradients
            sparse_vec = torch.log1p(weights.sum(dim=1))
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return sparse_vec

    def get_sparse_representation(
        self,
        sparse_vec: torch.Tensor,
        top_k: Optional[int] = None,
        threshold: float = 0.0
    ) -> List[Dict[int, float]]:
        """
        Convert dense sparse vector to dictionary format.

        Args:
            sparse_vec: [batch_size, vocab_size]
            top_k: Keep only top-k weights per vector
            threshold: Minimum weight to keep

        Returns:
            List of {token_id: weight} dictionaries
        """
        batch_size = sparse_vec.shape[0]
        results = []

        for i in range(batch_size):
            vec = sparse_vec[i]

            # Apply threshold
            if threshold > 0:
                vec = vec.clone()
                vec[vec < threshold] = 0

            # Get non-zero indices and values
            non_zero = vec.nonzero().squeeze(-1)
            values = vec[non_zero]

            # Apply top-k if specified
            if top_k and len(non_zero) > top_k:
                top_values, top_idx = values.topk(top_k)
                non_zero = non_zero[top_idx]
                values = top_values

            # Convert to dictionary
            results.append({
                idx.item(): val.item()
                for idx, val in zip(non_zero, values)
                if val.item() > 0
            })

        return results

    def flops_loss(
        self,
        query_vecs: torch.Tensor,
        doc_vecs: torch.Tensor,
        lambda_q: float = 0.001,
        lambda_d: float = 0.01
    ) -> torch.Tensor:
        """
        FLOPS regularization loss for sparsity control.

        Args:
            query_vecs: [batch, vocab] query sparse vectors
            doc_vecs: [batch, vocab] document sparse vectors
            lambda_q: Query sparsity weight
            lambda_d: Document sparsity weight

        Returns:
            Scalar loss
        """
        # Query: L2 norm of sum (encourages few non-zero terms)
        query_flops = (query_vecs.sum(dim=1) ** 2).mean()

        # Document: Sum of squares (encourages small weights)
        doc_flops = (doc_vecs ** 2).sum(dim=1).mean()

        return lambda_q * query_flops + lambda_d * doc_flops

    def contrastive_loss(
        self,
        query_vecs: torch.Tensor,
        pos_vecs: torch.Tensor,
        neg_vecs: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Contrastive loss for bi-encoder training.

        Args:
            query_vecs: [batch, vocab]
            pos_vecs: [batch, vocab]
            neg_vecs: [batch, num_neg, vocab]
            temperature: Softmax temperature

        Returns:
            Scalar loss
        """
        batch_size = query_vecs.shape[0]
        num_neg = neg_vecs.shape[1]

        # Positive scores: [batch]
        pos_scores = (query_vecs * pos_vecs).sum(dim=1) / temperature

        # Negative scores: [batch, num_neg]
        neg_scores = torch.einsum("bv,bnv->bn", query_vecs, neg_vecs) / temperature

        # Combine: [batch, 1 + num_neg]
        all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        # Labels: positive is at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=query_vecs.device)

        return F.cross_entropy(all_scores, labels)


class SPLADEEncoder:
    """
    High-level encoder for SPLADE inference.

    Handles tokenization, batching, and conversion to sparse format.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        tokenizer_name: str = "bert-base-uncased",
        device: str = "cuda",
        max_length: int = 256,
        aggregation: str = "max",
        vocab_size: int = 30522
    ):
        """
        Initialize SPLADE encoder.

        Args:
            model_path: Path to trained model weights (.pt or .bin)
            tokenizer_name: HuggingFace tokenizer name
            device: 'cuda' or 'cpu'
            max_length: Maximum sequence length
            aggregation: 'max', 'sum', or 'log1p_sum'
            vocab_size: Vocabulary size
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.aggregation = aggregation

        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

        # Load model
        self.model = SPLADEModel(
            model_name=tokenizer_name,
            aggregation=aggregation,
            vocab_size=vocab_size
        ).to(self.device)

        # Load weights
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        logger.info(f"Loaded SPLADE model from {model_path} on {self.device}")

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        top_k: Optional[int] = None,
        threshold: float = 0.0,
        show_progress: bool = False
    ) -> List[Dict[int, float]]:
        """
        Encode texts to sparse vectors.

        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            top_k: Keep top-k terms per vector
            threshold: Minimum weight threshold
            show_progress: Show progress bar

        Returns:
            List of {token_id: weight} dictionaries
        """
        from tqdm import tqdm

        all_sparse = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding")

        for i in iterator:
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            # Forward pass
            sparse_vecs = self.model(
                inputs["input_ids"],
                inputs["attention_mask"]
            )

            # Convert to sparse dict format
            sparse_dicts = self.model.get_sparse_representation(
                sparse_vecs,
                top_k=top_k,
                threshold=threshold
            )
            all_sparse.extend(sparse_dicts)

        return all_sparse

    @torch.no_grad()
    def encode_queries(
        self,
        queries: List[str],
        batch_size: int = 32,
        top_k: int = 100
    ) -> List[Dict[int, float]]:
        """Encode queries (typically fewer terms needed)."""
        return self.encode(queries, batch_size=batch_size, top_k=top_k)

    @torch.no_grad()
    def encode_documents(
        self,
        documents: List[str],
        batch_size: int = 16,
        top_k: int = 300
    ) -> List[Dict[int, float]]:
        """Encode documents (more terms for better recall)."""
        return self.encode(documents, batch_size=batch_size, top_k=top_k)

    def analyze_vector(self, sparse_vec: Dict[int, float], top_n: int = 20) -> Dict:
        """
        Analyze a sparse vector - show top terms and statistics.
        """
        sorted_terms = sorted(sparse_vec.items(), key=lambda x: x[1], reverse=True)

        return {
            "num_terms": len(sparse_vec),
            "total_weight": sum(sparse_vec.values()),
            "top_terms": [
                {
                    "token": self.tokenizer.decode([tid]),
                    "token_id": tid,
                    "weight": weight
                }
                for tid, weight in sorted_terms[:top_n]
            ],
            "weight_stats": {
                "max": max(sparse_vec.values()) if sparse_vec else 0,
                "mean": sum(sparse_vec.values()) / len(sparse_vec) if sparse_vec else 0,
                "min": min(sparse_vec.values()) if sparse_vec else 0
            }
        }


def load_pretrained_splade(
    model_name: str = "naver/splade-max",
    device: str = "cuda",
    max_length: int = 256
) -> SPLADEEncoder:
    """
    Load a pre-trained SPLADE model from HuggingFace Hub.

    Available models:
    - naver/splade-max (best quality)
    - naver/splade-distil (faster, good quality)
    - naver/splade-cocondenser
    - naver/multi-splade (multilingual)
    """
    from huggingface_hub import hf_hub_download

    # Download model weights
    model_path = hf_hub_download(
        repo_id=model_name,
        filename="pytorch_model.bin"
    )

    # Determine aggregation from model name
    aggregation = "max" if "max" in model_name or "distil" in model_name else "sum"

    return SPLADEEncoder(
        model_path=model_path,
        tokenizer_name="bert-base-uncased",
        device=device,
        max_length=max_length,
        aggregation=aggregation
    )