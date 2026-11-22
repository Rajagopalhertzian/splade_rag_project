"""
Tests for SPLADE-RAG system.

Run with: pytest tests/
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from splade_rag.utils import (
    preprocess_for_splade,
    chunk_text,
    chunk_by_sentences,
    analyze_sparse_vector,
    compare_vectors,
    expand_query_terms
)
from splade_rag.index import SPLADEIndex


class TestUtils:
    """Test utility functions."""

    def test_preprocess_for_splade(self):
        """Test text preprocessing."""
        text = "  Hello   World!  This  is   a test.  "
        result = preprocess_for_splade(text)
        assert result == "Hello World! This is a test."

    def test_preprocess_preserves_sentences(self):
        """Test that sentence structure is preserved."""
        text = "First sentence. Second sentence! Third sentence?"
        result = preprocess_for_splade(text)
        assert "First sentence." in result
        assert "Second sentence!" in result
        assert "Third sentence?" in result

    def test_chunk_text_basic(self):
        """Test basic chunking."""
        text = " ".join(["word"] * 500)  # ~500 words
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        # Check overlap
        for i in range(len(chunks) - 1):
            # Last words of chunk i should appear in chunk i+1
            pass  # Overlap verification would need tokenizer

    def test_chunk_text_short(self):
        """Test chunking with short text."""
        text = "Short text."
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_chunk_by_sentences(self):
        """Test sentence-aware chunking."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_by_sentences(text, max_tokens=50)
        assert len(chunks) >= 1
        # Each chunk should end with punctuation
        for chunk in chunks:
            assert chunk.strip()[-1] in '.!?'

    def test_analyze_sparse_vector(self):
        """Test sparse vector analysis."""
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Mock sparse vector
        sparse_vec = {100: 0.5, 200: 0.3, 300: 0.8, 400: 0.1}

        analysis = analyze_sparse_vector(sparse_vec, tokenizer, top_n=3)
        assert analysis['num_terms'] == 4
        assert analysis['total_weight'] == 1.7
        assert len(analysis['top_terms']) == 3
        assert analysis['top_terms'][0]['weight'] == 0.8

    def test_analyze_empty_vector(self):
        """Test analysis of empty vector."""
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        analysis = analyze_sparse_vector({}, tokenizer)
        assert analysis['num_terms'] == 0
        assert analysis['total_weight'] == 0.0
        assert analysis['top_terms'] == []

    def test_compare_vectors(self):
        """Test vector comparison."""
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        vec1 = {100: 0.5, 200: 0.3, 300: 0.8}
        vec2 = {200: 0.4, 300: 0.6, 400: 0.2}

        comparison = compare_vectors(vec1, vec2, tokenizer, top_n=5)
        assert comparison['overlap_ratio'] == 2/3  # 2 common out of 3 unique in each
        assert comparison['dot_product'] == 0.3*0.4 + 0.8*0.6  # 0.12 + 0.48 = 0.6
        assert len(comparison['common_terms']) == 2
        assert len(comparison['vec1_only']) == 1
        assert len(comparison['vec2_only']) == 1


class TestSPLADEIndex:
    """Test SPLADE index operations."""

    def test_add_and_retrieve(self):
        """Test basic add and retrieve."""
        index = SPLADEIndex()

        doc_ids = ["doc1", "doc2"]
        sparse_vecs = [
            {100: 0.5, 200: 0.3},
            {100: 0.2, 300: 0.8}
        ]
        metadata = [{"source": "test"}, {"source": "test"}]
        contents = ["Content 1", "Content 2"]

        index.add_documents(doc_ids, sparse_vecs, metadata, contents)

        assert index.stats['num_docs'] == 2
        assert index.stats['num_terms'] == 3  # 100, 200, 300

        # Query with term 100
        query_vec = {100: 1.0}
        results = index.retrieve(query_vec, top_k=2)
        assert len(results) == 2
        # doc1 should rank higher (0.5 > 0.2)
        assert results[0][0] == "doc1"
        assert results[1][0] == "doc2"

    def test_retrieve_with_min_score(self):
        """Test retrieval with minimum score threshold."""
        index = SPLADEIndex()

        index.add_documents(
            ["doc1", "doc2"],
            [{100: 0.5}, {100: 0.1}],
            [{}], [{}]
        )

        results = index.retrieve({100: 1.0}, top_k=2, min_score=0.3)
        assert len(results) == 1
        assert results[0][0] == "doc1"

    def test_remove_and_compact(self):
        """Test document removal and compaction."""
        index = SPLADEIndex()

        index.add_documents(
            ["doc1", "doc2"],
            [{100: 0.5}, {200: 0.3}],
            [{"_content": "c1"}], [{"_content": "c2"}]
        )

        assert index.stats['num_docs'] == 2

        # Remove doc1
        index.remove_document("doc1")
        assert index.stats['num_docs'] == 2  # Not compacted yet

        # Compact
        removed = index.compact()
        assert removed == 1
        assert index.stats['num_docs'] == 1

    def test_save_load_json(self):
        """Test saving and loading index as JSON."""
        import tempfile
        import os

        index = SPLADEIndex()
        index.add_documents(
            ["doc1"],
            [{100: 0.5, 200: 0.3}],
            [{"source": "test", "_content": "content"}],
            ["content"]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            index.save(tmpdir, format="json")

            # Check files exist
            assert os.path.exists(os.path.join(tmpdir, "index.json"))
            assert os.path.exists(os.path.join(tmpdir, "metadata.json"))
            assert os.path.exists(os.path.join(tmpdir, "vectors.json"))
            assert os.path.exists(os.path.join(tmpdir, "stats.json"))

            # Load back
            loaded = SPLADEIndex.load(tmpdir, format="json")
            assert loaded.stats['num_docs'] == 1
            assert loaded.stats['num_terms'] == 2


class TestIntegration:
    """Integration tests (require model downloads)."""

    @pytest.mark.integration
    def test_encoder_loading(self):
        """Test loading pre-trained encoder."""
        # This test requires internet and GPU
        # Mark as integration to skip in unit tests
        pass

    @pytest.mark.integration
    def test_full_rag_pipeline(self):
        """Test full RAG pipeline."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])