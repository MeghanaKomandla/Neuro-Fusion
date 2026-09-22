"""
chatbot.py — RAG Chatbot
=========================
ChromaDB (local vector search) + Gemini 1.5 Flash API

Flow:
  1. Load PDFs from knowledge_base/ folder
  2. Split into 500-word chunks
  3. Embed with all-MiniLM-L6-v2 (80MB, runs locally)
  4. Store in ChromaDB (persisted to chroma_db/ folder)
  5. On question: find top 3 relevant chunks
  6. Send chunks + question to Gemini API
  7. Return answer with source citations

Setup:
  1. Get free Gemini API key from aistudio.google.com
  2. Set environment variable: set GEMINI_API_KEY=your_key
  3. Add PDFs to knowledge_base/ folder
  4. First run builds the vector DB automatically
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PDF_DIR        = Path("knowledge_base")
CHROMA_DIR     = "chroma_db"

_vectordb  = None
_gemini    = None
_embeddings = None


def _get_embeddings():
    """Load sentence embeddings model — 80MB, runs on your PC."""
    global _embeddings
    if _embeddings is None:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            _embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("Embeddings model loaded")
        except Exception as e:
            logger.error(f"Embeddings failed: {e}")
    return _embeddings


def _get_vectordb():
    """Load or build ChromaDB from PDFs in knowledge_base/."""
    global _vectordb
    if _vectordb is not None:
        return _vectordb

    embeddings = _get_embeddings()
    if embeddings is None:
        return None

    try:
        from langchain_community.vectorstores import Chroma

        # Load existing DB if built already
        if Path(CHROMA_DIR).exists() and any(Path(CHROMA_DIR).iterdir()):
            _vectordb = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings,
            )
            logger.info(f"ChromaDB loaded from {CHROMA_DIR}")
            return _vectordb

        # Build from PDFs
        PDF_DIR.mkdir(exist_ok=True)
        pdfs = list(PDF_DIR.glob("*.pdf"))

        if not pdfs:
            logger.warning(
                "No PDFs found in knowledge_base/. "
                "Add research papers to get cited answers.")
            return None

        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        docs = []
        for pdf in pdfs:
            try:
                loader = PyPDFLoader(str(pdf))
                docs.extend(loader.load())
                logger.info(f"Loaded: {pdf.name}")
            except Exception as e:
                logger.warning(f"Could not load {pdf.name}: {e}")

        if not docs:
            return None

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        logger.info(f"Built {len(chunks)} chunks from {len(pdfs)} PDFs")

        _vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
        )
        logger.info("ChromaDB built and saved")
        return _vectordb

    except Exception as e:
        logger.error(f"ChromaDB failed: {e}")
        return None


def _get_gemini():
    """Initialize Gemini API client using new google-genai package."""
    global _gemini
    if _gemini is not None:
        return _gemini
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        logger.error(
            "GEMINI_API_KEY not set. "
            "Get free key from aistudio.google.com "
            "then run: set GEMINI_API_KEY=your_key")
        return None
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        _gemini = client
        logger.info("Gemini API initialized (google-genai)")
        return _gemini
    except Exception as e:
        logger.error(f"Gemini init failed: {e}")
        return None
    
def ask_chatbot(question: str) -> dict:
    """
    RAG + Gemini pipeline.
    Returns dict with answer and sources.
    """
    if not question or not question.strip():
        return dict(answer="Please ask a question.", sources=[])

    gemini = _get_gemini()
    if gemini is None:
        return dict(
            answer=(
                "Chatbot unavailable: GEMINI_API_KEY not set. "
                "Get a free key from aistudio.google.com and "
                "set it with: set GEMINI_API_KEY=your_key_here"),
            sources=[])

    db = _get_vectordb()

    if db is not None:
        # RAG: find relevant research chunks
        try:
            relevant = db.similarity_search(question, k=3)
            context  = "\n\n".join([d.page_content for d in relevant])
            sources  = list(set([
                Path(d.metadata.get("source","Unknown")).name
                for d in relevant
            ]))
        except Exception:
            context = ""
            sources = []
    else:
        context = ""
        sources = ["General knowledge (add PDFs to knowledge_base/ for citations)"]

    # Build prompt
    if context:
        prompt = f"""You are a medical research assistant specializing in 
Alzheimer's disease and Dementia. Answer using ONLY the research below.
If the answer is not in the research, clearly say so.

Research:
{context}

Question: {question}

Give a clear, helpful answer and mention the source document."""
    else:
        prompt = f"""You are a medical research assistant specializing in
Alzheimer's disease and Dementia. Answer this question clearly and accurately.
Note: No research documents are loaded yet.

Question: {question}"""

    try:
        response = gemini.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
        )
        return dict(answer=response.text, sources=sources)
    except Exception as e:
        return dict(
            answer=f"Error calling Gemini API: {e}",
            sources=[])


def rebuild_knowledge_base() -> dict:
    """Force rebuild the ChromaDB from PDFs."""
    global _vectordb
    import shutil
    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)
    _vectordb = None
    db = _get_vectordb()
    pdfs = list(PDF_DIR.glob("*.pdf"))
    return dict(
        success=db is not None,
        pdfs_loaded=len(pdfs),
        message=f"Rebuilt from {len(pdfs)} PDFs" if db else "No PDFs found",
    )
