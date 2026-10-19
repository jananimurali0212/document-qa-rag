"""
Document Q&A RAG Studio & Evaluation Harness.

A modern, production-grade Streamlit application for:
- Context-grounded document question answering with verifiable citations.
- Secure environment-driven configuration (no UI API key inputs).
- Document ingestion and knowledge base lifecycle management.
- Deterministic retrieval and generation benchmarking (Hit Rate@K, Recall@K, MRR, Precision@K).
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load environment variables automatically from .env
load_dotenv()

# Internal modules
from src.ingestion.file_validator import FileValidator, IngestionStatus
from src.ingestion.document_parser import DocumentParser, ParsingError
from src.ingestion.chunker import Chunker, ChunkRecord
from src.embeddings.embedding_service import EmbeddingService
from src.retrieval.vector_store import VectorStore, SearchResult
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker
from src.generation.prompt_builder import PromptBuilder
from src.generation.answer_generator import AnswerGenerator
from src.evaluation.dataset_loader import DatasetLoader, BenchmarkQuestion
from src.evaluation.retrieval_metrics import evaluate_single_query
from src.evaluation.answer_metrics import AnswerMetricsEvaluator
from src.evaluation.benchmark_runner import BenchmarkRunner, BenchmarkSummary
from src.evaluation.report_generator import ReportGenerator
from src.storage.metadata_store import MetadataStore


# ==========================================
# SECURE CONFIGURATION RESOLVER
# ==========================================
def get_gemini_api_key() -> str:
    """
    Securely load Gemini API key exclusively from Streamlit secrets or environment variables.
    Zero manual input required in the UI.
    """
    api_key = ""
    # 1. Streamlit secrets
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            api_key = str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass

    # 2. OS Environment variable / .env file
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()

    return api_key


# ==========================================
# PAGE CONFIGURATION & MODERN DESIGN SYSTEM
# ==========================================
st.set_page_config(
    page_title="RAG Studio • Document Q&A & Evaluation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    /* Hero Header */
    .hero-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
        color: #FFFFFF;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.15);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        background: linear-gradient(90deg, #60A5FA 0%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 0.92rem;
        color: #94A3B8;
        margin-top: 6px;
        max-width: 650px;
        line-height: 1.45;
    }

    /* Status Pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .status-pill-online {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border-color: rgba(52, 211, 153, 0.3);
    }
    .status-pill-offline {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border-color: rgba(251, 191, 36, 0.3);
    }

    /* Modern Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 14px;
        margin: 16px 0 24px 0;
    }
    .modern-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .modern-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px -4px rgba(0, 0, 0, 0.08);
        border-color: #CBD5E1;
    }
    .card-label {
        font-size: 0.76rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .card-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        margin: 6px 0 2px 0;
        line-height: 1.1;
    }
    .card-target {
        font-size: 0.74rem;
        color: #94A3B8;
        font-weight: 500;
    }

    /* Citation Cards */
    .citation-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #3B82F6;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 10px 0;
        transition: background-color 0.15s ease;
    }
    .citation-card:hover {
        background: #F1F5F9;
    }
    .citation-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
    }
    .citation-badge {
        font-size: 0.74rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
        background: #DBEAFE;
        color: #1E40AF;
    }
    .citation-text {
        font-size: 0.86rem;
        color: #334155;
        line-height: 1.5;
        font-style: italic;
    }

    /* Quick Prompt Chips */
    .chip-container {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin: 10px 0 16px 0;
    }

    /* Sidebar Clean Styling */
    .sidebar-section {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
    /* Hide Streamlit Deploy Button, Toolbar Actions & Decoration */
    .stDeployButton,
    [data-testid="stDeployButton"],
    .stAppDeployButton,
    button[title*="Deploy"],
    button[kind="header"]:has(svg),
    div[data-testid="stToolbarActions"],
    header [data-testid="stToolbarActions"],
    header div:has(> button[title*="Deploy"]),
    #stDecoration,
    #MainMenu {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================
# STATE INITIALIZATION
# ==========================================
def init_session_state():
    """Initialize persistent components in Streamlit session state."""
    if "metadata_store" not in st.session_state:
        st.session_state.metadata_store = MetadataStore(db_path="rag_metadata.db")

    if "embedding_service" not in st.session_state:
        st.session_state.embedding_service = EmbeddingService(
            model_name="all-MiniLM-L6-v2",
            use_fallback_if_unavailable=True
        )

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore(
            dimension=st.session_state.embedding_service.dimension,
            use_faiss=True
        )

    if "reranker" not in st.session_state:
        st.session_state.reranker = Reranker(use_neural=False)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "benchmark_summary" not in st.session_state:
        st.session_state.benchmark_summary = None

    if "k_sweep_results" not in st.session_state:
        st.session_state.k_sweep_results = None


init_session_state()

# Detect API Key securely from environment or Streamlit secrets
gemini_key = get_gemini_api_key()
is_llm_connected = bool(gemini_key)


# ==========================================
# STREAMLINED SIDEBAR (NO API INPUT FIELDS)
# ==========================================
with st.sidebar:
    st.markdown("### ⚡ RAG Studio")
    st.caption("Document Question Answering & Retrieval Evaluation")

    # Security & Connectivity Badge
    if is_llm_connected:
        st.markdown(
            '<div class="status-pill status-pill-online">🟢 Gemini LLM Connected</div>',
            unsafe_allow_html=True
        )
        st.caption("🔐 Authenticated via `.env` / system environment.")
    else:
        st.markdown(
            '<div class="status-pill status-pill-offline">🟡 Local Grounded Extractor</div>',
            unsafe_allow_html=True
        )
        st.caption("ℹ️ Running in zero-dependency local mode. Add `GEMINI_API_KEY` to `.env` to enable generative synthesis.")

    st.divider()

    # Knowledge Base Overview
    store_stats = st.session_state.vector_store.get_stats()
    st.markdown("**Knowledge Base Index**")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Documents", store_stats["total_documents"])
    with col_s2:
        st.metric("Chunks", store_stats["total_chunks"])

    st.caption(f"**Index Backend:** `{store_stats['backend']}`")
    st.caption(f"**Embedder:** `{'Deterministic Fallback' if st.session_state.embedding_service.is_fallback else 'Sentence Transformers'}`")

    st.divider()

    # Clean Advanced Tuning Expander
    with st.expander("⚙️ Advanced Pipeline Tuning", expanded=False):
        top_k = st.slider("Top-K Candidates", min_value=1, max_value=10, value=5, step=1, help="Number of retrieved chunks for generation.")
        enable_reranker = st.toggle("Enable Reranker", value=True, help="Applies lexical-semantic fusion to optimize candidate ranking.")
        chunk_size = st.slider("Chunk Size", min_value=150, max_value=1200, value=500, step=50, help="Target character size for text chunks.")
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=min(250, chunk_size - 50), value=100, step=25)

        st.write("")
        if st.button("🧹 Reset Knowledge Base", use_container_width=True, type="secondary"):
            st.session_state.vector_store.clear()
            st.session_state.metadata_store.clear_all()
            st.session_state.chat_history.clear()
            st.success("Knowledge base cleared.")
            time.sleep(0.6)
            st.rerun()


# Runtime components initialized cleanly with environment parameters
retriever = Retriever(
    embedding_service=st.session_state.embedding_service,
    vector_store=st.session_state.vector_store,
    reranker=st.session_state.reranker if enable_reranker else None
)

answer_generator = AnswerGenerator(
    api_key=gemini_key,
    model_name="gemini-3.8-flash",
    prompt_builder=PromptBuilder()
)


# ==========================================
# HERO BANNER
# ==========================================
st.markdown(
    f"""
    <div class="hero-container">
        <div>
            <h1 class="hero-title">Document Q&A RAG Studio</h1>
            <p class="hero-subtitle">
                Context-grounded natural-language question answering with verifiable source citations, 
                coupled with a deterministic evaluation harness for measuring retrieval accuracy and hallucination risks.
            </p>
        </div>
        <div style="text-align: right;">
            <span class="status-pill {'status-pill-online' if is_llm_connected else 'status-pill-offline'}">
                {'🟢 Gemini Online' if is_llm_connected else '⚡ Local Engine'}
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# MAIN TAB NAVIGATION
# ==========================================
tab_chat, tab_ingest, tab_eval, tab_reports = st.tabs([
    "💬 Chat & Q&A",
    "📁 Document Upload & Corpus",
    "📊 Evaluation Dashboard",
    "📑 Reports & Audit History"
])


# ==========================================
# TAB 1: CHAT & CONTEXT-GROUNDED Q&A
# ==========================================
with tab_chat:
    col_chat_hdr, col_chat_ctrl = st.columns([3, 1])
    with col_chat_hdr:
        st.subheader("Interactive Document Q&A")
        st.caption("Ask questions about your uploaded documents. Answers strictly require verified supporting citations.")
    with col_chat_ctrl:
        if st.button("🔄 Clear Conversation", use_container_width=True):
            st.session_state.chat_history.clear()
            st.rerun()

    # Pre-flight check: if vector store is empty, prompt sample indexing
    all_docs = st.session_state.metadata_store.get_all_documents()
    if not all_docs:
        st.info("💡 **Quick Start**: You haven't uploaded any documents yet. You can upload files in the **'Document Upload'** tab, or click below to load a sample technical corpus.", icon="📄")
        if st.button("🚀 Load Sample Technical Guide (Instant Demo)", type="primary"):
            sample_corpus = [
                "The Document Q&A Chatbot enables users to upload PDF, DOCX, and TXT files. It uses Retrieval-Augmented Generation to answer questions with verifiable citations.",
                "The system architecture combines document parsing, chunking, dense vector embeddings, top-K retrieval, and LLM answer generation with strict abstention rules.",
                "File validation enforces support for PDF (PyMuPDF), DOCX (python-docx), and TXT formats, rejecting corrupt files and enforcing file size constraints.",
                "The chunking engine splits parsed text into overlapping passages with chunk_id, document_id, document_name, page_number, text, and chunk_index.",
                "Deterministic retrieval metrics include Hit Rate@K, Recall@K, MRR, and Precision@K calculated against annotated ground-truth chunk labels.",
                "Mean Reciprocal Rank (MRR) evaluates how early the first relevant result appears, computed as the mean of 1/r across all benchmark questions.",
                "Faithfulness measures whether answer claims are strictly supported by the retrieved context passages, helping diagnose hallucination failures.",
                "When retrieved context is insufficient to answer the question, the prompt instructs the model to abstain rather than invent unsupported facts."
            ]
            chunker = Chunker(chunk_size=300, chunk_overlap=50)
            sample_chunks = chunker.chunk_text_blocks(sample_corpus, document_id="doc01", document_name="Technical_Architecture_Guide.pdf")
            embs = st.session_state.embedding_service.embed_chunks(sample_chunks)
            st.session_state.vector_store.add_chunks(sample_chunks, embs)
            st.session_state.metadata_store.record_document("doc01", "Technical_Architecture_Guide.pdf", ".pdf", 12500, 1, len(sample_chunks))
            st.session_state.metadata_store.record_chunks(sample_chunks)
            st.success("Sample knowledge base loaded! Try asking a question below.")
            time.sleep(0.5)
            st.rerun()

    # Render Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander(f"📌 Verified Citations ({len(msg['citations'])} sources)", expanded=False):
                    for cite in msg["citations"]:
                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <div class="citation-header">
                                    <span><strong>📄 {cite['document']}</strong> • Page {cite['page']}</span>
                                    <span class="citation-badge">Chunk: {cite['chunk_id']}</span>
                                </div>
                                <div class="citation-text">"{cite['snippet']}"</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    # Chat Input
    query_input = st.chat_input("Ask a question about your documents...")
    if query_input:
        st.session_state.chat_history.append({"role": "user", "content": query_input})
        with st.chat_message("user"):
            st.markdown(query_input)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving verified passages and synthesizing answer..."):
                retrieved_chunks = retriever.retrieve(query_input, top_k=top_k)
                result = answer_generator.generate_answer(query_input, retrieved_chunks)

                st.markdown(result.answer)

                citations_payload = []
                if result.citations:
                    citations_payload = [
                        {
                            "document": c.document_name,
                            "page": c.page_number,
                            "chunk_id": c.chunk_id,
                            "snippet": c.snippet,
                            "score": c.score
                        }
                        for c in result.citations
                    ]
                    with st.expander(f"📌 Verified Citations ({len(citations_payload)} sources)", expanded=True):
                        for cite in citations_payload:
                            st.markdown(
                                f"""
                                <div class="citation-card">
                                    <div class="citation-header">
                                        <span><strong>📄 {cite['document']}</strong> • Page {cite['page']}</span>
                                        <span class="citation-badge">Score: {cite['score']:.2f} | Chunk: {cite['chunk_id']}</span>
                                    </div>
                                    <div class="citation-text">"{cite['snippet']}"</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                st.caption(f"⚡ Latency: {result.latency_ms:.1f}ms • Model: {result.model_name} • Top-K: {top_k}")

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result.answer,
                    "citations": citations_payload
                })


# ==========================================
# TAB 2: DOCUMENT UPLOAD & CORPUS MANAGEMENT
# ==========================================
with tab_ingest:
    st.subheader("Document Ingestion & Knowledge Base")
    st.caption("Upload files to parse, chunk, embed, and index into the vector database.")

    col_upload_box, col_upload_btn = st.columns([3, 1])
    with col_upload_box:
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, or TXT documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="PyMuPDF preserves PDF page numbering. python-docx extracts DOCX paragraphs and tables."
        )
    with col_upload_btn:
        st.write("")
        st.write("")
        ingest_action = st.button("🚀 Process & Ingest", type="primary", use_container_width=True)

    if ingest_action and uploaded_files:
        validator = FileValidator()
        parser = DocumentParser()
        chunker_instance = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        current_docs = st.session_state.metadata_store.get_all_documents()
        doc_count = len(current_docs)

        progress = st.progress(0, text="Initializing ingestion...")
        for idx, file_obj in enumerate(uploaded_files):
            file_bytes = file_obj.getvalue()
            doc_counter = doc_count + idx + 1

            # 1. Validation
            val_res = validator.validate_file(
                file_bytes,
                filename=file_obj.name,
                sequential_counter=doc_counter
            )

            if not val_res.is_valid:
                st.error(f"❌ '{file_obj.name}' rejected: {val_res.error_message}")
                continue

            # 2. Parsing
            try:
                parsed_doc = parser.parse(file_bytes, val_res.document_id, val_res.document_name)
            except Exception as e:
                st.error(f"❌ Failed to parse '{file_obj.name}': {str(e)}")
                continue

            # 3. Chunking
            chunks = chunker_instance.chunk_document(parsed_doc)
            if not chunks:
                st.warning(f"⚠️ Document '{file_obj.name}' produced no extractable text chunks.")
                continue

            # 4. Embeddings & Indexing
            embeddings = st.session_state.embedding_service.embed_chunks(chunks)
            st.session_state.vector_store.add_chunks(chunks, embeddings)

            # 5. Metadata Store Recording
            st.session_state.metadata_store.record_document(
                document_id=val_res.document_id,
                document_name=val_res.document_name,
                file_type=val_res.file_type,
                file_size_bytes=val_res.file_size_bytes,
                num_pages=parsed_doc.total_pages,
                num_chunks=len(chunks),
                status=IngestionStatus.INDEXED
            )
            st.session_state.metadata_store.record_chunks(chunks)
            progress.progress((idx + 1) / len(uploaded_files), text=f"Processed: {file_obj.name}")

        st.success("🎉 Documents successfully indexed into the knowledge base!")
        time.sleep(0.8)
        st.rerun()

    st.divider()

    # Active Document Inventory Table
    active_docs = st.session_state.metadata_store.get_all_documents()
    if active_docs:
        st.markdown(f"#### 📑 Ingested Document Inventory ({len(active_docs)} files)")
        df_docs = pd.DataFrame(active_docs)

        st.dataframe(
            df_docs,
            column_config={
                "document_id": st.column_config.TextColumn("Doc ID", width="small"),
                "document_name": st.column_config.TextColumn("File Name", width="medium"),
                "file_type": st.column_config.TextColumn("Format", width="small"),
                "num_pages": st.column_config.NumberColumn("Pages", width="small"),
                "num_chunks": st.column_config.NumberColumn("Chunks", width="small"),
                "file_size_bytes": st.column_config.NumberColumn("Bytes", format="%d B"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "created_at": st.column_config.DatetimeColumn("Ingested At", format="YYYY-MM-DD HH:mm")
            },
            hide_index=True,
            use_container_width=True
        )

        col_delete_doc, col_inspect_chunk = st.columns([1, 2])
        with col_delete_doc:
            doc_to_delete = st.selectbox(
                "Delete Document",
                options=[d["document_id"] for d in active_docs],
                format_func=lambda x: f"{x} - {next((d['document_name'] for d in active_docs if d['document_id'] == x), '')}"
            )
            if st.button("🗑️ Remove Document", type="secondary", use_container_width=True):
                st.session_state.vector_store.delete_document(doc_to_delete)
                st.session_state.metadata_store.delete_document(doc_to_delete)
                st.success(f"Document `{doc_to_delete}` and its vector embeddings were removed.")
                time.sleep(0.6)
                st.rerun()

        with col_inspect_chunk:
            selected_inspect_id = st.selectbox(
                "Inspect Chunks for Document",
                options=[d["document_id"] for d in active_docs],
                key="chunk_inspect_box"
            )
            if selected_inspect_id:
                inspect_chunks = st.session_state.metadata_store.get_chunks_for_document(selected_inspect_id)
                with st.expander(f"🔍 Inspect Chunks ({len(inspect_chunks)} total)"):
                    for c in inspect_chunks[:8]:
                        st.markdown(f"**Chunk ID:** `{c['chunk_id']}` • **Page {c['page_number']}** • Index {c['chunk_index']}")
                        st.code(c["text"], language="text")
                    if len(inspect_chunks) > 8:
                        st.caption(f"Showing first 8 of {len(inspect_chunks)} chunks.")
    else:
        st.info("No documents uploaded yet. Upload your files above to populate the knowledge base.")


# ==========================================
# TAB 3: EVALUATION BENCHMARK DASHBOARD
# ==========================================
with tab_eval:
    st.subheader("RAG Evaluation Benchmark Harness")
    st.caption("Measure retrieval success and generation quality against a ground-truth golden dataset.")

    dataset_path = "evaluation_data/evaluation_dataset.json"

    col_eval_info, col_eval_ctrl = st.columns([2, 1])
    with col_eval_info:
        if Path(dataset_path).exists():
            benchmark_questions = DatasetLoader.load_from_file(dataset_path)
            st.success(f"✅ Golden Dataset Loaded: **{len(benchmark_questions)} annotated questions** (`{dataset_path}`)", icon="📁")
        else:
            st.error("Evaluation dataset missing at evaluation_data/evaluation_dataset.json")
            benchmark_questions = []

    with col_eval_ctrl:
        eval_k = st.selectbox("Benchmark Top-K Rank", options=[1, 3, 5, 10], index=2)
        run_eval_btn = st.button("⚡ Run Evaluation Benchmark", type="primary", use_container_width=True)

    if run_eval_btn and benchmark_questions:
        # Auto-index sample corpus if store is empty
        stats = st.session_state.vector_store.get_stats()
        if stats["total_chunks"] == 0:
            st.warning("⚠️ Vector store was empty. Auto-indexing sample technical documentation...")
            sample_corpus = [
                "The Document Q&A Chatbot enables users to upload PDF, DOCX, and TXT files. It uses Retrieval-Augmented Generation to answer questions with verifiable citations.",
                "The system architecture combines document parsing, chunking, dense vector embeddings, top-K retrieval, and LLM answer generation with strict abstention rules.",
                "File validation enforces support for PDF (PyMuPDF), DOCX (python-docx), and TXT formats, rejecting corrupt files and enforcing file size constraints.",
                "The chunking engine splits parsed text into overlapping passages with chunk_id, document_id, document_name, page_number, text, and chunk_index.",
                "Deterministic retrieval metrics include Hit Rate@K, Recall@K, MRR, and Precision@K calculated against annotated ground-truth chunk labels.",
                "Mean Reciprocal Rank (MRR) evaluates how early the first relevant result appears, computed as the mean of 1/r across all benchmark questions.",
                "Faithfulness measures whether answer claims are strictly supported by the retrieved context passages, helping diagnose hallucination failures.",
                "When retrieved context is insufficient to answer the question, the prompt instructs the model to abstain rather than invent unsupported facts."
            ]
            chunker = Chunker(chunk_size=300, chunk_overlap=50)
            sample_chunks = chunker.chunk_text_blocks(sample_corpus, document_id="doc01", document_name="project_spec.pdf")
            embs = st.session_state.embedding_service.embed_chunks(sample_chunks)
            st.session_state.vector_store.add_chunks(sample_chunks, embs)
            st.session_state.metadata_store.record_document("doc01", "project_spec.pdf", ".pdf", 5000, 1, len(sample_chunks))
            st.session_state.metadata_store.record_chunks(sample_chunks)

        with st.spinner(f"Running deterministic retrieval and answer quality evaluation (K={eval_k})..."):
            runner = BenchmarkRunner(
                retriever=retriever,
                answer_generator=answer_generator,
                answer_evaluator=AnswerMetricsEvaluator()
            )
            summary = runner.run_benchmark(benchmark_questions, k=eval_k, generate_answers=True)
            st.session_state.benchmark_summary = summary

            # Run parameter sweep K = 1, 3, 5
            k_sweep = runner.run_k_sweep(benchmark_questions, k_values=(1, 3, 5), generate_answers=False)
            st.session_state.k_sweep_results = k_sweep

            # Record experiment in SQLite metadata store
            st.session_state.metadata_store.record_experiment(
                experiment_id=summary.experiment_id,
                k=summary.k,
                num_questions=summary.num_questions,
                hit_rate_at_k=summary.hit_rate_at_k,
                recall_at_k=summary.recall_at_k,
                mrr=summary.mrr,
                precision_at_k=summary.precision_at_k,
                faithfulness=summary.mean_faithfulness,
                avg_latency_ms=summary.avg_latency_ms,
                summary_dict=summary.to_dict()
            )

    # Render Benchmark Summary if Available
    if st.session_state.benchmark_summary is not None:
        summary: BenchmarkSummary = st.session_state.benchmark_summary
        st.markdown("---")
        st.markdown(f"### 🎯 Benchmark Summary • Experiment `{summary.experiment_id}`")

        # 6 KPI Tiles
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.markdown(
                f"""
                <div class="modern-card">
                    <div class="card-label">Hit Rate@{summary.k}</div>
                    <div class="card-val">{summary.hit_rate_at_k*100:.1f}%</div>
                    <div class="card-target">Target: ≥ 80.0%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f"""
                <div class="modern-card">
                    <div class="card-label">Recall@{summary.k}</div>
                    <div class="card-val">{summary.recall_at_k:.3f}</div>
                    <div class="card-target">Target: ≥ 0.700</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c3:
            st.markdown(
                f"""
                <div class="modern-card">
                    <div class="card-label">MRR</div>
                    <div class="card-val">{summary.mrr:.3f}</div>
                    <div class="card-target">Target: ≥ 0.700</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c4:
            st.markdown(
                f"""
                <div class="modern-card">
                    <div class="card-label">Precision@{summary.k}</div>
                    <div class="card-val">{summary.precision_at_k:.3f}</div>
                    <div class="card-target">Dense retrieval</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c5:
            faith_text = f"{summary.mean_faithfulness:.2f}" if summary.mean_faithfulness is not None else "N/A"
            st.markdown(
                f"""
                <div class="modern-card">
                    <div class="card-label">Faithfulness</div>
                    <div class="card-val">{faith_text}</div>
                    <div class="card-target">Target: ≥ 0.85</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c6:
            st.markdown(
                f"""
                <div class="modern-card">
                    <div class="card-label">Avg Latency</div>
                    <div class="card-val">{summary.avg_latency_ms:.0f}ms</div>
                    <div class="card-target">Target: < 1500ms</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        # Parameter Sweep Charts (K = 1, 3, 5)
        if st.session_state.k_sweep_results:
            st.markdown("#### 📈 Parameter Sensitivity: Top-K Trade-Offs (K = 1 vs 3 vs 5)")
            sweep_data = []
            for k_v, s_v in st.session_state.k_sweep_results.items():
                sweep_data.append({
                    "Rank Cutoff": f"K = {k_v}",
                    "Hit Rate": round(s_v.hit_rate_at_k, 3),
                    "Recall": round(s_v.recall_at_k, 3),
                    "MRR": round(s_v.mrr, 3),
                    "Precision": round(s_v.precision_at_k, 3),
                })
            df_sweep = pd.DataFrame(sweep_data)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_sweep["Rank Cutoff"], y=df_sweep["Hit Rate"], name="Hit Rate@K", marker_color="#2563EB"))
            fig.add_trace(go.Bar(x=df_sweep["Rank Cutoff"], y=df_sweep["Recall"], name="Recall@K", marker_color="#059669"))
            fig.add_trace(go.Bar(x=df_sweep["Rank Cutoff"], y=df_sweep["MRR"], name="MRR", marker_color="#7C3AED"))
            fig.add_trace(go.Bar(x=df_sweep["Rank Cutoff"], y=df_sweep["Precision"], name="Precision@K", marker_color="#D97706"))
            fig.update_layout(
                barmode="group",
                height=340,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Per-Question Results Table with Rich Formatter
        st.markdown("#### 📋 Detailed Per-Question Evaluation Breakdown")
        df_results = ReportGenerator.to_dataframe(summary)
        st.dataframe(
            df_results,
            column_config={
                "Question": st.column_config.TextColumn("ID", width="small"),
                "Query": st.column_config.TextColumn("Benchmark Query", width="large"),
                "Hit@K": st.column_config.NumberColumn("Hit", format="%d"),
                "Recall@K": st.column_config.ProgressColumn("Recall@K", min_value=0.0, max_value=1.0, format="%.2f"),
                "MRR": st.column_config.NumberColumn("MRR", format="%.2f"),
                "Precision@K": st.column_config.NumberColumn("Precision@K", format="%.2f"),
                "Faithfulness": st.column_config.TextColumn("Faithfulness", width="small"),
                "Latency_ms": st.column_config.NumberColumn("Latency (ms)", format="%.1f ms"),
            },
            hide_index=True,
            use_container_width=True
        )


# ==========================================
# TAB 4: REPORTS & AUDIT HISTORY
# ==========================================
with tab_reports:
    st.subheader("Benchmark Reports & Experiment History")
    st.caption("Export reproducible evaluation runs as CSV and JSON or review historical experiments stored in SQLite.")

    if st.session_state.benchmark_summary is not None:
        summary: BenchmarkSummary = st.session_state.benchmark_summary

        col_csv, col_json = st.columns(2)
        with col_csv:
            csv_data = ReportGenerator.export_csv(summary)
            st.download_button(
                label="📥 Download Benchmark Results (CSV)",
                data=csv_data,
                file_name=f"benchmark_{summary.experiment_id}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_json:
            json_data = ReportGenerator.export_json(summary)
            st.download_button(
                label="📥 Download Benchmark Summary (JSON)",
                data=json_data,
                file_name=f"benchmark_{summary.experiment_id}.json",
                mime="application/json",
                use_container_width=True
            )

        with st.expander("📄 View Markdown Executive Report", expanded=False):
            md_text = ReportGenerator.generate_markdown_report(summary)
            st.markdown(md_text)
    else:
        st.info("Run an evaluation benchmark in the **'Evaluation Dashboard'** tab to generate downloadable reports.")

    st.divider()

    # Past Experiments from SQLite
    past_experiments = st.session_state.metadata_store.get_experiments()
    if past_experiments:
        st.markdown(f"#### 🏛️ Experiment Run History ({len(past_experiments)} runs)")
        df_exp = pd.DataFrame(past_experiments)
        st.dataframe(
            df_exp,
            column_config={
                "experiment_id": st.column_config.TextColumn("Experiment ID", width="medium"),
                "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm"),
                "k": st.column_config.NumberColumn("K", width="small"),
                "num_questions": st.column_config.NumberColumn("Queries", width="small"),
                "hit_rate_at_k": st.column_config.NumberColumn("Hit Rate@K", format="%.3f"),
                "recall_at_k": st.column_config.NumberColumn("Recall@K", format="%.3f"),
                "mrr": st.column_config.NumberColumn("MRR", format="%.3f"),
                "precision_at_k": st.column_config.NumberColumn("Precision@K", format="%.3f"),
                "faithfulness": st.column_config.NumberColumn("Faithfulness", format="%.2f"),
                "avg_latency_ms": st.column_config.NumberColumn("Avg Latency", format="%.0f ms"),
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No saved experiments in SQLite database yet.")
