#!/usr/bin/env python
"""
Streamlit Web UI for SPLADE-RAG.

Run with: streamlit run scripts/streamlit_app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from splade_rag import SPLADERAG, RAGConfig, create_rag_system
from splade_rag.encoder import SPLADEEncoder
from splade_rag.index import SPLADEIndex

# Page config
st.set_page_config(
    page_title="SPLADE-RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stAlert > div { padding: 0.5rem 1rem; }
    .source-box {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_system(config_dict):
    """Load and cache the RAG system."""
    config = RAGConfig(**config_dict)
    return SPLADERAG(config)


@st.cache_resource
def load_encoder_and_index(model_path, index_path, device):
    """Load encoder and index for analysis."""
    encoder = SPLADEEncoder(model_path=model_path, device=device)
    index = SPLADEIndex.load(index_path)
    return encoder, index


def main():
    st.title("🔍 SPLADE-RAG: Sparse Retrieval Augmented Generation")
    st.markdown("""
    A production-ready RAG system using **SPLADE (Sparse Lexical and Expansion Model)**
    for efficient, interpretable sparse retrieval combined with LLM generation.
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model selection
        splade_model = st.selectbox(
            "SPLADE Model",
            ["naver/splade-max", "naver/splade-distil", "naver/splade-cocondenser"],
            index=0,
            help="naver/splade-max: Best quality | naver/splade-distil: Faster"
        )

        llm_model = st.selectbox(
            "LLM Model",
            ["microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3-medium-4k-instruct",
             "meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"],
            index=0
        )

        index_path = st.text_input(
            "Index Path",
            value="./data/splade_index",
            help="Path to saved SPLADE index"
        )

        device = st.selectbox("Device", ["cuda", "cpu"], index=0 if st.session_state.get("cuda_available", False) else 1)

        # Retrieval settings
        st.subheader("Retrieval Settings")
        retrieval_top_k = st.slider("First-stage Top-K", 10, 200, 100)
        final_top_k = st.slider("Final Top-K", 1, 20, 5)
        use_reranker = st.checkbox("Use Re-ranker", value=True)
        reranker_model = st.text_input("Re-ranker Model", "cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Generation settings
        st.subheader("Generation Settings")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
        max_tokens = st.slider("Max New Tokens", 128, 2048, 512)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.1)

        # Load button
        if st.button("🔄 Load System", type="primary"):
            with st.spinner("Loading SPLADE-RAG system..."):
                try:
                    config_dict = {
                        "splade_model_path": splade_model,
                        "llm_model_name": llm_model,
                        "device": device,
                        "index_path": index_path,
                        "retrieval_top_k": retrieval_top_k,
                        "final_top_k": final_top_k,
                        "use_reranker": use_reranker,
                        "reranker_model": reranker_model,
                        "temperature": temperature,
                        "max_new_tokens": max_tokens,
                        "top_p": top_p
                    }
                    st.session_state.rag = load_rag_system(config_dict)
                    st.session_state.config = config_dict
                    st.success("✅ System loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to load: {e}")

    # Check if system is loaded
    if "rag" not in st.session_state:
        st.info("👈 Configure and load the system from the sidebar to start querying.")
        return

    rag = st.session_state.rag

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Query", "🔍 Retrieval Analysis", "📊 Index Stats", "📚 Document Management"])

    with tab1:
        st.header("Ask a Question")

        # Query input
        question = st.text_area(
            "Your Question",
            placeholder="Enter your question here...",
            height=100
        )

        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            query_top_k = st.number_input("Top-K", 1, 20, final_top_k)
        with col2:
            stream_response = st.checkbox("Stream Response", value=True)

        if st.button("🚀 Get Answer", type="primary", disabled=not question):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Retrieving and generating..."):
                    try:
                        if stream_response:
                            # Streaming response
                            answer_placeholder = st.empty()
                            full_answer = ""
                            for token in rag.stream_query(question, top_k=query_top_k):
                                full_answer += token
                                answer_placeholder.markdown(full_answer + "▌")
                            answer_placeholder.markdown(full_answer)

                            # Get citations separately
                            result = rag.query(question, top_k=query_top_k)
                            citations = result.citations
                        else:
                            result = rag.query(question, top_k=query_top_k)
                            st.markdown(result.answer)
                            citations = result.citations

                        # Display citations
                        if citations:
                            st.subheader("📚 Sources")
                            for cite in citations:
                                with st.expander(f"[Source {cite['source_id']}] {cite['doc_id']} (score: {cite['score']:.4f})"):
                                    meta = cite.get('metadata', {})
                                    content = meta.get('_content', '')
                                    if content:
                                        st.text(content[:500] + ("..." if len(content) > 500 else ""))
                                    st.json({k: v for k, v in meta.items() if k != '_content'})

                    except Exception as e:
                        st.error(f"Error: {e}")

    with tab2:
        st.header("Retrieval Analysis")
        st.markdown("Analyze how SPLADE expands queries and matches documents.")

        analysis_question = st.text_input("Query to analyze", placeholder="Enter a query...")

        if analysis_question and st.button("Analyze"):
            with st.spinner("Analyzing..."):
                try:
                    encoder = rag.encoder
                    query_vec = encoder.encode_queries([analysis_question])[0]

                    # Show query expansion
                    analysis = encoder.analyze_vector(query_vec, top_n=30)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Active Terms", analysis['num_terms'])
                        st.metric("Total Weight", f"{analysis['total_weight']:.2f}")

                        st.subheader("Top Query Terms")
                        for term in analysis['top_terms']:
                            st.write(f"**{term['token']}**: {term['weight']:.4f}")

                    with col2:
                        # Retrieve and show matching docs
                        results = rag.pipeline.search(analysis_question, final_top_k=5)

                        st.subheader("Top Retrieved Documents")
                        for i, doc in enumerate(results):
                            with st.expander(f"#{i+1} {doc['doc_id']} (score: {doc.get('score', doc.get('fused_score', doc.get('rerank_score', 0))):.4f})"):
                                content = doc.get('content', doc.get('_content', ''))
                                st.text(content[:300] + "...")

                                # Show term overlap
                                if 'content' in doc or '_content' in doc:
                                    doc_vec = rag.index.get_document_vector(doc['doc_id'])
                                    if doc_vec:
                                        from splade_rag.utils import compare_vectors
                                        comparison = compare_vectors(query_vec, doc_vec, encoder.tokenizer, top_n=10)
                                        st.write("**Common Terms:**")
                                        for token, weight in comparison['common_terms']:
                                            st.write(f"  {token}: {weight:.4f}")

                except Exception as e:
                    st.error(f"Analysis error: {e}")

    with tab3:
        st.header("Index Statistics")

        if st.button("🔄 Refresh Stats"):
            stats = rag.get_stats()
            st.session_state.index_stats = stats

        if "index_stats" in st.session_state:
            stats = st.session_state.index_stats
            idx_stats = stats.get('index', {})

            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Documents", f"{idx_stats.get('num_docs', 0):,}")
            with col2:
                st.metric("Vocabulary Terms", f"{idx_stats.get('num_terms', 0):,}")
            with col3:
                st.metric("Total Postings", f"{idx_stats.get('total_postings', 0):,}")
            with col4:
                st.metric("Avg Terms/Doc", f"{idx_stats.get('avg_terms_per_doc', 0):.1f}")

            # Config
            st.subheader("System Configuration")
            config = stats.get('config', {})
            for key, value in config.items():
                st.write(f"**{key}**: {value}")

    with tab4:
        st.header("Document Management")

        st.subheader("Add Documents")

        # File upload
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['txt', 'md', 'json'],
            accept_multiple_files=True
        )

        if uploaded_files and st.button("Add to Index"):
            with st.spinner("Processing and indexing documents..."):
                try:
                    documents = []
                    for file in uploaded_files:
                        content = file.read().decode('utf-8')
                        documents.append({
                            "id": file.name,
                            "content": content,
                            "metadata": {"source_file": file.name, "uploaded": True}
                        })

                    rag.add_documents(documents, save_index=True)
                    st.success(f"✅ Added {len(documents)} documents to index!")

                    # Refresh stats
                    stats = rag.get_stats()
                    st.session_state.index_stats = stats

                except Exception as e:
                    st.error(f"Error adding documents: {e}")

        # Directory indexing
        st.subheader("Index from Directory")
        dir_path = st.text_input("Directory Path", placeholder="./data/documents")
        extensions = st.multiselect(
            "File Extensions",
            [".txt", ".md", ".json", ".pdf", ".html"],
            default=[".txt", ".md", ".json"]
        )

        if dir_path and st.button("Build Index from Directory"):
            with st.spinner("Building index from directory..."):
                try:
                    from splade_rag.utils import load_documents_from_directory
                    docs = load_documents_from_directory(dir_path, extensions=extensions)
                    if docs:
                        rag.add_documents(docs, save_index=True)
                        st.success(f"✅ Indexed {len(docs)} documents from {dir_path}")
                        stats = rag.get_stats()
                        st.session_state.index_stats = stats
                    else:
                        st.warning("No documents found in directory.")
                except Exception as e:
                    st.error(f"Error: {e}")

        # Save index
        st.subheader("Save Index")
        save_path = st.text_input("Save Path", value=index_path)
        if st.button("💾 Save Index"):
            try:
                rag.save_index(save_path)
                st.success(f"✅ Index saved to {save_path}")
            except Exception as e:
                st.error(f"Error saving: {e}")


if __name__ == "__main__":
    # Check CUDA availability
    import torch
    st.session_state.cuda_available = torch.cuda.is_available()
    main()