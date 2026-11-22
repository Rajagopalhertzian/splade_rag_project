#!/usr/bin/env python
"""
Build SPLADE index from documents.

Usage:
    python scripts/build_index.py --data_dir ./data/documents --output ./data/splade_index
    python scripts/build_index.py --config config/config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from splade_rag import SPLADEEncoder, SPLADEIndex, create_index_from_documents
from splade_rag.utils import load_documents_from_directory, chunk_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build SPLADE index from documents")
    parser.add_argument("--data_dir", type=str, help="Directory containing documents")
    parser.add_argument("--output", type=str, help="Output index directory")
    parser.add_argument("--config", type=str, help="Path to config.yaml")
    parser.add_argument("--model", type=str, default="naver/splade-max", help="SPLADE model")
    parser.add_argument("--chunk_size", type=int, default=200, help="Chunk size in tokens")
    parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap in tokens")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--extensions", nargs="+", default=[".txt", ".md", ".json"],
                        help="File extensions to index")

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        data_dir = config.get("data", {}).get("documents_dir", args.data_dir)
        output = config.get("index", {}).get("path", args.output)
        model = config.get("model", {}).get("splade_model", args.model)
        chunk_size = config.get("chunking", {}).get("chunk_size", args.chunk_size)
        overlap = config.get("chunking", {}).get("chunk_overlap", args.overlap)
        device = config.get("model", {}).get("device", args.device)
        extensions = config.get("data", {}).get("supported_extensions", args.extensions)
    else:
        data_dir = args.data_dir
        output = args.output
        model = args.model
        chunk_size = args.chunk_size
        overlap = args.overlap
        device = args.device
        extensions = args.extensions

    if not data_dir or not output:
        parser.error("--data_dir and --output required (or use --config)")

    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    # Load documents
    logger.info(f"Loading documents from {data_dir}...")
    raw_docs = load_documents_from_directory(data_dir, extensions=extensions, recursive=True)

    if not raw_docs:
        logger.warning("No documents found!")
        sys.exit(0)

    logger.info(f"Loaded {len(raw_docs)} documents")

    # Chunk documents
    all_chunks = []
    chunk_metadata = []

    for doc in raw_docs:
        chunks = chunk_text(
            doc["content"],
            chunk_size=chunk_size,
            overlap=overlap
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

    logger.info(f"Created {len(all_chunks)} chunks")

    # Initialize encoder
    logger.info(f"Loading SPLADE model: {model}")
    encoder = SPLADEEncoder(
        model_path=model,
        device=device
    )

    # Create index
    logger.info("Building index...")
    index = SPLADEIndex()

    # Encode and add in batches
    batch_size = 16
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i+batch_size]
        batch_meta = chunk_metadata[i:i+batch_size]
        batch_ids = [m["source_doc_id"] + f"_chunk_{m['chunk_index']}" for m in batch_meta]

        sparse_vecs = encoder.encode_documents(batch_chunks)
        index.add_documents(batch_ids, sparse_vecs, batch_meta, batch_chunks)

        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"  Processed {i + len(batch_chunks)}/{len(all_chunks)} chunks")

    # Save index
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    index.save(output_path, format="json")

    logger.info(f"Index built and saved to {output}")
    logger.info(f"Stats: {index.stats}")


if __name__ == "__main__":
    main()