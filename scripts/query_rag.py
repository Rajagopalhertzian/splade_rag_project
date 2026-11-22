#!/usr/bin/env python
"""
Interactive SPLADE-RAG query script.

Usage:
    python scripts/query_rag.py --index ./data/splade_index
    python scripts/query_rag.py --config config/config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from splade_rag import SPLADERAG, RAGConfig, create_rag_system
from splade_rag.encoder import load_pretrained_splade

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Query SPLADE-RAG system")
    parser.add_argument("--index", type=str, help="Path to SPLADE index")
    parser.add_argument("--config", type=str, help="Path to config.yaml")
    parser.add_argument("--model", type=str, default="naver/splade-max", help="SPLADE model")
    parser.add_argument("--llm", type=str, default="microsoft/Phi-3-mini-4k-instruct", help="LLM model")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results")
    parser.add_argument("--no_rerank", action="store_true", help="Disable re-ranking")
    parser.add_argument("--stream", action="store_true", help="Stream response")
    parser.add_argument("--question", type=str, help="Single question (non-interactive)")

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)

        rag_config = RAGConfig(
            splade_model_path=config.get("model", {}).get("splade_model", args.model),
            llm_model_name=config.get("model", {}).get("llm_model", args.llm),
            device=config.get("model", {}).get("device", args.device),
            index_path=config.get("index", {}).get("path", args.index),
            retrieval_top_k=config.get("retrieval", {}).get("top_k_candidates", 100),
            final_top_k=config.get("retrieval", {}).get("final_top_k", args.top_k),
            use_reranker=config.get("reranker", {}).get("enabled", not args.no_rerank),
            reranker_model=config.get("reranker", {}).get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            max_new_tokens=config.get("generation", {}).get("max_new_tokens", 512),
            temperature=config.get("generation", {}).get("temperature", 0.3),
            top_p=config.get("generation", {}).get("top_p", 0.9)
        )
        rag = SPLADERAG(rag_config)
    else:
        if not args.index:
            parser.error("--index required (or use --config)")
        rag = create_rag_system(
            splade_model=args.model,
            llm_model=args.llm,
            index_path=args.index,
            device=args.device,
            final_top_k=args.top_k,
            use_reranker=not args.no_rerank
        )

    # Single question mode
    if args.question:
        result = rag.query(args.question)
        print(f"\nQuestion: {args.question}")
        print(f"\nAnswer: {result.answer}")
        print(f"\nSources ({len(result.citations)}):")
        for cite in result.citations:
            print(f"  [{cite['source_id']}] {cite['doc_id']} (score: {cite['score']:.4f})")
        return

    # Interactive mode
    print("\n" + "="*60)
    print("SPLADE-RAG Interactive Query")
    print("Type 'quit' or 'exit' to quit")
    print("Type 'stats' for system statistics")
    print("="*60 + "\n")

    while True:
        try:
            question = input("\nQuestion: ").strip()

            if question.lower() in ('quit', 'exit', 'q'):
                break

            if question.lower() == 'stats':
                stats = rag.get_stats()
                print(f"\nIndex Stats: {stats['index']}")
                print(f"Config: {stats['config']}")
                continue

            if not question:
                continue

            print("\nRetrieving...")
            if args.stream:
                print("Answer: ", end="", flush=True)
                for token in rag.stream_query(question):
                    print(token, end="", flush=True)
                print()
            else:
                result = rag.query(question)
                print(f"\nAnswer: {result.answer}")
                print(f"\nSources ({len(result.citations)}):")
                for cite in result.citations:
                    print(f"  [{cite['source_id']}] {cite['doc_id']} (score: {cite['score']:.4f})")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Error: {e}")


if __name__ == "__main__":
    main()