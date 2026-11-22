#!/usr/bin/env python
"""
Evaluate SPLADE-RAG system on benchmark datasets.

Usage:
    python scripts/evaluate.py --index ./data/splade_index --test_data ./data/test.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from splade_rag import SPLADERAG, RAGConfig, create_rag_system
from splade_rag.encoder import SPLADEEncoder
from splade_rag.index import SPLADEIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_test_data(path: str) -> List[Dict]:
    """Load test data in JSONL format."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def evaluate_retrieval(
    encoder: SPLADEEncoder,
    index: SPLADEIndex,
    test_data: List[Dict],
    top_k: int = 100
) -> Dict[str, float]:
    """
    Evaluate retrieval metrics: Recall@k, MRR@k, nDCG@k.
    """
    from splade_rag.retriever import SPLADERetriever

    retriever = SPLADERetriever(index, encoder)

    all_recall = {k: [] for k in [1, 5, 10, 20, 50, 100]}
    all_mrr = {k: [] for k in [10, 100]}
    all_ndcg = {k: [] for k in [10, 100]}

    logger.info(f"Evaluating retrieval on {len(test_data)} queries...")

    for item in tqdm(test_data, desc="Evaluating"):
        query = item["query"]
        relevant_docs = set(item.get("relevant_docs", []))

        if not relevant_docs:
            continue

        # Retrieve
        results = retriever.retrieve(query, top_k=top_k)
        retrieved_ids = [r["doc_id"] for r in results]

        # Calculate metrics
        for k in all_recall.keys():
            retrieved_k = set(retrieved_ids[:k])
            recall = len(retrieved_k & relevant_docs) / len(relevant_docs)
            all_recall[k].append(recall)

        for k in all_mrr.keys():
            mrr = 0.0
            for rank, doc_id in enumerate(retrieved_ids[:k], 1):
                if doc_id in relevant_docs:
                    mrr = 1.0 / rank
                    break
            all_mrr[k].append(mrr)

        for k in all_ndcg.keys():
            # Simple nDCG calculation
            dcg = 0.0
            for rank, doc_id in enumerate(retrieved_ids[:k], 1):
                if doc_id in relevant_docs:
                    dcg += 1.0 / np.log2(rank + 1)

            # Ideal DCG
            ideal_rels = min(len(relevant_docs), k)
            idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_rels + 1))

            ndcg = dcg / idcg if idcg > 0 else 0.0
            all_ndcg[k].append(ndcg)

    # Aggregate
    metrics = {}
    for k, values in all_recall.items():
        metrics[f"recall@{k}"] = np.mean(values) if values else 0.0
    for k, values in all_mrr.items():
        metrics[f"mrr@{k}"] = np.mean(values) if values else 0.0
    for k, values in all_ndcg.items():
        metrics[f"ndcg@{k}"] = np.mean(values) if values else 0.0

    return metrics


def evaluate_rag(
    rag: SPLADERAG,
    test_data: List[Dict],
    top_k: int = 5
) -> Dict[str, float]:
    """
    Evaluate end-to-end RAG quality.
    Uses LLM-as-judge or simple heuristics.
    """
    logger.info(f"Evaluating RAG on {len(test_data)} questions...")

    correct = 0
    total = 0
    citation_precision = []
    citation_recall = []

    for item in tqdm(test_data, desc="RAG Evaluation"):
        question = item["question"]
        expected_answer = item.get("answer", "")
        expected_sources = set(item.get("sources", []))

        result = rag.query(question, top_k=top_k)

        # Simple heuristic: check if expected answer keywords appear
        if expected_answer:
            answer_lower = result.answer.lower()
            expected_lower = expected_answer.lower()

            # Keyword overlap
            expected_words = set(expected_lower.split())
            answer_words = set(answer_lower.split())
            overlap = len(expected_words & answer_words) / max(len(expected_words), 1)

            if overlap > 0.3:  # Threshold
                correct += 1
            total += 1

        # Citation quality
        cited_sources = set(c["doc_id"] for c in result.citations)
        if expected_sources:
            prec = len(cited_sources & expected_sources) / max(len(cited_sources), 1)
            rec = len(cited_sources & expected_sources) / max(len(expected_sources), 1)
            citation_precision.append(prec)
            citation_recall.append(rec)

    metrics = {}
    if total > 0:
        metrics["answer_accuracy"] = correct / total
    if citation_precision:
        metrics["citation_precision"] = np.mean(citation_precision)
        metrics["citation_recall"] = np.mean(citation_recall)
        metrics["citation_f1"] = 2 * metrics["citation_precision"] * metrics["citation_recall"] / \
                                 max(metrics["citation_precision"] + metrics["citation_recall"], 1e-8)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate SPLADE-RAG system")
    parser.add_argument("--index", type=str, required=True, help="Path to SPLADE index")
    parser.add_argument("--test_data", type=str, required=True, help="Test data (JSONL)")
    parser.add_argument("--model", type=str, default="naver/splade-max", help="SPLADE model")
    parser.add_argument("--llm", type=str, default="microsoft/Phi-3-mini-4k-instruct", help="LLM model")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--top_k", type=int, default=100, help="Retrieval top-k")
    parser.add_argument("--final_k", type=int, default=5, help="Final top-k for RAG")
    parser.add_argument("--output", type=str, help="Output results file")
    parser.add_argument("--eval_retrieval", action="store_true", help="Evaluate retrieval only")
    parser.add_argument("--eval_rag", action="store_true", help="Evaluate full RAG")

    args = parser.parse_args()

    if not args.eval_retrieval and not args.eval_rag:
        args.eval_retrieval = True
        args.eval_rag = True

    # Load test data
    test_data = load_test_data(args.test_data)
    logger.info(f"Loaded {len(test_data)} test examples")

    results = {}

    # Retrieval evaluation
    if args.eval_retrieval:
        logger.info("Loading encoder and index for retrieval evaluation...")
        encoder = SPLADEEncoder(
            model_path=args.model,
            device=args.device
        )
        index = SPLADEIndex.load(args.index)

        retrieval_metrics = evaluate_retrieval(encoder, index, test_data, args.top_k)
        results["retrieval"] = retrieval_metrics

        logger.info("Retrieval Metrics:")
        for k, v in retrieval_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

    # RAG evaluation
    if args.eval_rag:
        logger.info("Loading full RAG system...")
        rag = create_rag_system(
            splade_model=args.model,
            llm_model=args.llm,
            index_path=args.index,
            device=args.device,
            final_top_k=args.final_k
        )

        rag_metrics = evaluate_rag(rag, test_data, args.final_k)
        results["rag"] = rag_metrics

        logger.info("RAG Metrics:")
        for k, v in rag_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")

    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    for category, metrics in results.items():
        print(f"\n{category.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    from tqdm import tqdm
    main()