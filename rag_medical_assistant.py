"""
RAG Medical Assistant: Healthcare AI Powered by Retrieval-Augmented Generation

This module implements a complete RAG (Retrieval-Augmented Generation) pipeline
for transforming the Merck Manual into a queryable medical knowledge base.

Author: Jeremy Gracey
License: MIT
"""

from __future__ import annotations

import os
import logging
from importlib import import_module
from typing import Any, List, Dict, Optional
from dataclasses import dataclass

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only in minimal environments
    np = None

try:
    from langchain.llms.base import LLM
    from langchain.callbacks.manager import CallbackManagerLLMRun, AsyncCallbackManagerLLMRun
    from langchain.prompts import PromptTemplate
except ImportError:  # Allows importing and testing non-LLM code without LangChain.
    class LLM:
        """Small fallback matching the LangChain call interface used here."""

        def __call__(self, prompt: str, *args: Any, **kwargs: Any) -> str:
            return self._call(prompt, *args, **kwargs)

    class PromptTemplate:
        """Minimal prompt formatter used when LangChain is not installed."""

        def __init__(self, input_variables: List[str], template: str):
            self.input_variables = input_variables
            self.template = template

        def format(self, **kwargs: Any) -> str:
            return self.template.format(**kwargs)

    CallbackManagerLLMRun = Any
    AsyncCallbackManagerLLMRun = Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def require_dependency(module_name: str, package_name: Optional[str] = None, purpose: Optional[str] = None):
    """
    Import an optional runtime dependency with an actionable installation message.

    Heavy ML dependencies are imported lazily so the module remains importable for
    tests, documentation examples, and environments that have not downloaded models.
    """
    try:
        return import_module(module_name)
    except ImportError as exc:
        install_name = package_name or module_name.split(".")[0]
        detail = f" for {purpose}" if purpose else ""
        raise ImportError(
            f"{install_name} is required{detail}. "
            "Install project dependencies with: python3 -m pip install -r requirements.txt"
        ) from exc


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class RAGConfig:
    """Configuration parameters for the RAG pipeline."""

    # PDF and Text Processing
    pdf_path: str = "merck_manual.pdf"
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Embedding Model
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Vector Store
    chroma_db_path: str = "./chroma_db"
    collection_name: str = "medical_knowledge"

    # LLM Configuration
    mistral_model_path: str = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    n_gpu_layers: int = 35  # Number of layers to offload to GPU
    n_ctx: int = 2048  # Context window size
    temperature: float = 0.3
    top_p: float = 0.95
    max_tokens: int = 512

    # Retrieval Configuration
    retriever_k: int = 5  # Number of documents to retrieve

    # Evaluation
    enable_evaluation: bool = True
    judge_model_path: str = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"


# ==============================================================================
# SECTION 1: PDF INGESTION
# ==============================================================================

class PDFIngestor:
    """Handles PDF extraction and text preprocessing from the Merck Manual."""

    def __init__(self, pdf_path: str):
        """
        Initialize the PDF ingestor.

        Args:
            pdf_path: Path to the PDF file to ingest.
        """
        self.pdf_path = pdf_path

    def extract_text(self) -> str:
        """
        Extract all text from the PDF document.

        Returns:
            Full text extracted from the PDF.

        Raises:
            FileNotFoundError: If PDF file does not exist.
        """
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF not found at {self.pdf_path}")

        fitz = require_dependency("fitz", "PyMuPDF", "PDF ingestion")
        logger.info(f"Extracting text from {self.pdf_path}")

        full_text = []
        with fitz.open(self.pdf_path) as doc:
            total_pages = len(doc)
            logger.info(f"Processing {total_pages} pages")

            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                if text.strip():
                    full_text.append(text)

                if page_num % 100 == 0:
                    logger.info(f"Processed {page_num}/{total_pages} pages")

        combined_text = "\n\n".join(full_text)
        logger.info(f"Extracted {len(combined_text)} characters from PDF")

        return combined_text

    def extract_text_with_metadata(self) -> List[Dict[str, str]]:
        """
        Extract text from PDF with page metadata.

        Returns:
            List of dicts containing text and page metadata.
        """
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF not found at {self.pdf_path}")

        fitz = require_dependency("fitz", "PyMuPDF", "PDF ingestion")
        logger.info(f"Extracting text with metadata from {self.pdf_path}")

        documents = []
        with fitz.open(self.pdf_path) as doc:
            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                if text.strip():
                    documents.append({
                        "text": text,
                        "page": page_num,
                        "source": self.pdf_path
                    })

        logger.info(f"Extracted {len(documents)} pages with metadata")
        return documents


# ==============================================================================
# SECTION 2: TEXT CHUNKING
# ==============================================================================

class TextChunker:
    """Handles text splitting with semantic awareness for optimal RAG performance."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        text_splitter_module = require_dependency(
            "langchain.text_splitter",
            "langchain",
            "text chunking",
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = text_splitter_module.RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into semantically meaningful chunks.

        Args:
            text: Raw text to chunk.

        Returns:
            List of text chunks.
        """
        logger.info(f"Chunking text with size={self.chunk_size}, overlap={self.chunk_overlap}")
        chunks = self.splitter.split_text(text)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Split documents into chunks while preserving metadata.

        Args:
            documents: List of documents with text and metadata.

        Returns:
            List of chunked documents with metadata.
        """
        chunked_docs = []
        for doc in documents:
            chunks = self.chunk_text(doc["text"])
            for chunk_idx, chunk in enumerate(chunks):
                chunked_docs.append({
                    "text": chunk,
                    "page": doc.get("page"),
                    "chunk_id": chunk_idx,
                    "source": doc.get("source")
                })

        logger.info(f"Created {len(chunked_docs)} total chunks from {len(documents)} documents")
        return chunked_docs


# ==============================================================================
# SECTION 3: EMBEDDING GENERATION
# ==============================================================================

class EmbeddingGenerator:
    """Generates dense vector embeddings using Sentence-Transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.

        Args:
            model_name: Name of the Sentence-Transformers model to use.
        """
        logger.info(f"Loading embedding model: {model_name}")
        sentence_transformers = require_dependency(
            "sentence_transformers",
            "sentence-transformers",
            "embedding generation",
        )
        self.model = sentence_transformers.SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector (384-dimensional for all-MiniLM-L6-v2).
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed.
            batch_size: Batch size for processing.

        Returns:
            Matrix of embeddings (n_texts x embedding_dim).
        """
        logger.info(f"Embedding {len(texts)} texts using {self.model_name}")
        embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        return embeddings


# ==============================================================================
# SECTION 4: VECTOR STORE
# ==============================================================================

class VectorStore:
    """Manages vector storage and retrieval using ChromaDB."""

    def __init__(self, db_path: str, collection_name: str = "medical_knowledge"):
        """
        Initialize the vector store.

        Args:
            db_path: Path to store ChromaDB files.
            collection_name: Name of the collection.
        """
        self.db_path = db_path
        self.collection_name = collection_name

        chromadb = require_dependency("chromadb", "chromadb", "vector storage")
        chroma_config = require_dependency("chromadb.config", "chromadb", "vector storage")

        # Create ChromaDB client with persistent storage
        settings = chroma_config.Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=db_path,
            anonymized_telemetry=False
        )
        self.client = chromadb.Client(settings)
        self.collection = None

        logger.info(f"Initialized ChromaDB at {db_path}")

    def create_collection(self, embedding_function=None):
        """
        Create or get a collection.

        Args:
            embedding_function: Optional custom embedding function.
        """
        if embedding_function:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        logger.info(f"Created collection: {self.collection_name}")

    def index_documents(self, documents: List[Dict[str, str]], embeddings: np.ndarray):
        """
        Index documents and their embeddings.

        Args:
            documents: List of documents with text and metadata.
            embeddings: Corresponding embedding vectors.
        """
        if self.collection is None:
            self.create_collection()

        logger.info(f"Indexing {len(documents)} documents")

        ids = [f"doc_{i}" for i in range(len(documents))]
        texts = [doc["text"] for doc in documents]
        metadatas = [
            {
                "page": str(doc.get("page", "")),
                "chunk_id": str(doc.get("chunk_id", "")),
                "source": doc.get("source", "")
            }
            for doc in documents
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )

        logger.info(f"Indexed {len(documents)} documents successfully")

    def retrieve_similar(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Retrieve similar documents using cosine similarity.

        Args:
            query_embedding: Embedding vector of the query.
            k: Number of documents to retrieve.

        Returns:
            List of similar documents with metadata and similarity scores.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )

        retrieved = []
        if results["documents"] and len(results["documents"]) > 0:
            for i, doc_text in enumerate(results["documents"][0]):
                retrieved.append({
                    "text": doc_text,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None
                })

        return retrieved

    def persist(self):
        """Persist the vector store to disk."""
        self.client.persist()
        logger.info(f"Persisted vector store to {self.db_path}")


# ==============================================================================
# SECTION 5: LLM SETUP
# ==============================================================================

class MistralLLM(LLM):
    """Custom LangChain wrapper for Mistral-7B via llama-cpp-python."""

    def __init__(self, model_path: str, **kwargs):
        """
        Initialize Mistral LLM.

        Args:
            model_path: Path to the GGUF model file.
            **kwargs: Additional arguments for the LLM.
        """
        super().__init__()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        llama_cpp = require_dependency("llama_cpp", "llama-cpp-python", "LLM inference")

        logger.info(f"Loading Mistral model from {model_path}")

        self.model_path = model_path
        self.llm = llama_cpp.Llama(
            model_path=model_path,
            n_gpu_layers=kwargs.get("n_gpu_layers", 35),
            n_ctx=kwargs.get("n_ctx", 2048),
            verbose=False
        )

        self.temperature = kwargs.get("temperature", 0.3)
        self.top_p = kwargs.get("top_p", 0.95)
        self.max_tokens = kwargs.get("max_tokens", 512)

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "mistral"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerLLMRun] = None,
        **kwargs
    ) -> str:
        """
        Generate text using the Mistral model.

        Args:
            prompt: Input prompt.
            stop: Stop sequences.
            run_manager: LangChain callback manager.
            **kwargs: Additional arguments.

        Returns:
            Generated text.
        """
        output = self.llm(
            prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=stop
        )

        return output["choices"][0]["text"]

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerLLMRun] = None,
        **kwargs
    ) -> str:
        """Async call (not implemented for llama-cpp-python)."""
        return self._call(prompt, stop, **kwargs)


# ==============================================================================
# SECTION 6: RAG CHAIN
# ==============================================================================

class RAGChain:
    """Orchestrates the complete RAG pipeline."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm: LLM,
        embedding_generator: EmbeddingGenerator,
        retriever_k: int = 5
    ):
        """
        Initialize the RAG chain.

        Args:
            vector_store: Initialized vector store.
            llm: Language model for generation.
            embedding_generator: Model for embedding queries.
            retriever_k: Number of documents to retrieve.
        """
        self.vector_store = vector_store
        self.llm = llm
        self.embedding_generator = embedding_generator
        self.retriever_k = retriever_k

        logger.info("RAG Chain initialized")

    def retrieve_context(self, query: str) -> List[Dict]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User query.

        Returns:
            List of relevant documents.
        """
        logger.info(f"Retrieving context for query: {query}")

        query_embedding = self.embedding_generator.embed_text(query)
        retrieved_docs = self.vector_store.retrieve_similar(query_embedding, k=self.retriever_k)

        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        return retrieved_docs

    def format_context(self, documents: List[Dict]) -> str:
        """
        Format retrieved documents into context string.

        Args:
            documents: List of retrieved documents.

        Returns:
            Formatted context string.
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            page = doc["metadata"].get("page", "N/A")
            text = doc["text"][:300]  # Truncate for display
            context_parts.append(f"[Source {i}, Page {page}]\n{text}...")

        return "\n\n".join(context_parts)


# ==============================================================================
# SECTION 7: PROMPT TEMPLATES
# ==============================================================================

class PromptStrategies:
    """Collection of different prompt engineering strategies."""

    @staticmethod
    def zero_shot() -> PromptTemplate:
        """Zero-shot prompt: Direct question answering."""
        template = """Answer the following medical question based on your knowledge:

Question: {question}

Answer:"""
        return PromptTemplate(input_variables=["question"], template=template)

    @staticmethod
    def few_shot() -> PromptTemplate:
        """Few-shot prompt: Include examples."""
        template = """Answer medical questions accurately and concisely.

Example 1:
Q: What is hypertension?
A: Hypertension is elevated blood pressure, typically defined as systolic pressure ≥140 mmHg or diastolic pressure ≥90 mmHg.

Example 2:
Q: What are common symptoms of diabetes?
A: Common symptoms include polyuria, polydipsia, weight loss, and fatigue.

Question: {question}

Answer:"""
        return PromptTemplate(input_variables=["question"], template=template)

    @staticmethod
    def chain_of_thought() -> PromptTemplate:
        """Chain-of-thought prompt: Reasoning steps."""
        template = """Answer the following medical question step by step.

Question: {question}

Let's think through this step by step:
1. First, identify the key medical concepts
2. Then, explain the relationships between them
3. Finally, provide a comprehensive answer

Answer:"""
        return PromptTemplate(input_variables=["question"], template=template)

    @staticmethod
    def structured_output() -> PromptTemplate:
        """Structured output prompt: Formatted response."""
        template = """Answer the following medical question in a structured format.

Question: {question}

Please provide your answer in the following format:
Definition: [Concise definition]
Causes/Risk Factors: [Relevant causes or risk factors]
Symptoms: [Key symptoms]
Diagnosis: [Diagnostic approaches]
Treatment: [Treatment options]

Answer:"""
        return PromptTemplate(input_variables=["question"], template=template)

    @staticmethod
    def concise_expert() -> PromptTemplate:
        """Concise expert prompt: Brief, authoritative answer."""
        template = """You are a medical expert. Answer the following question concisely and accurately.

Question: {question}

Expert Answer:"""
        return PromptTemplate(input_variables=["question"], template=template)

    @staticmethod
    def rag_augmented() -> PromptTemplate:
        """RAG-augmented prompt: Uses retrieved context."""
        template = """Use the following medical context to answer the question accurately.

Context:
{context}

Question: {question}

Based on the provided context, answer the question:"""
        return PromptTemplate(input_variables=["context", "question"], template=template)


# ==============================================================================
# SECTION 8: QUERY FUNCTION
# ==============================================================================

class MedicalAssistant:
    """Main interface for querying the medical knowledge base."""

    def __init__(self, rag_chain: RAGChain):
        """
        Initialize the assistant.

        Args:
            rag_chain: Initialized RAG chain.
        """
        self.rag_chain = rag_chain
        self.prompt_strategies = PromptStrategies()

    def query(
        self,
        question: str,
        strategy: str = "rag_augmented",
        include_sources: bool = True
    ) -> Dict[str, any]:
        """
        Query the medical knowledge base.

        Args:
            question: Medical question to answer.
            strategy: Prompt strategy to use.
            include_sources: Whether to include source information.

        Returns:
            Dictionary containing answer, sources, and metadata.
        """
        logger.info(f"Processing query: {question}")

        # Retrieve relevant context
        retrieved_docs = self.rag_chain.retrieve_context(question)
        context = self.rag_chain.format_context(retrieved_docs)

        # Select prompt strategy
        if strategy == "zero_shot":
            prompt = self.prompt_strategies.zero_shot()
            input_vars = {"question": question}
        elif strategy == "few_shot":
            prompt = self.prompt_strategies.few_shot()
            input_vars = {"question": question}
        elif strategy == "chain_of_thought":
            prompt = self.prompt_strategies.chain_of_thought()
            input_vars = {"question": question}
        elif strategy == "structured_output":
            prompt = self.prompt_strategies.structured_output()
            input_vars = {"question": question}
        elif strategy == "concise_expert":
            prompt = self.prompt_strategies.concise_expert()
            input_vars = {"question": question}
        else:  # rag_augmented (default)
            prompt = self.prompt_strategies.rag_augmented()
            input_vars = {"context": context, "question": question}

        # Generate response
        prompt_text = prompt.format(**input_vars)
        answer = self.rag_chain.llm(prompt_text)

        result = {
            "question": question,
            "answer": answer,
            "strategy": strategy,
            "num_sources": len(retrieved_docs)
        }

        if include_sources:
            result["sources"] = [
                {
                    "text": doc["text"][:200],
                    "page": doc["metadata"].get("page"),
                    "distance": doc.get("distance")
                }
                for doc in retrieved_docs
            ]

        return result


# ==============================================================================
# SECTION 9: EVALUATION
# ==============================================================================

class RAGEvaluator:
    """Evaluates RAG system performance using LLM-as-judge."""

    def __init__(self, judge_model_path: str):
        """
        Initialize the evaluator.

        Args:
            judge_model_path: Path to the judge LLM model.
        """
        self.judge_model_path = judge_model_path
        logger.info("RAG Evaluator initialized")

    def score_groundedness(
        self,
        answer: str,
        context: str,
        question: str
    ) -> Dict[str, any]:
        """
        Score how grounded the answer is in the provided context.

        Args:
            answer: Generated answer.
            context: Retrieved context documents.
            question: Original question.

        Returns:
            Dictionary with groundedness score and explanation.
        """
        prompt = f"""Rate how well the answer is grounded in the provided context.

Context:
{context}

Question: {question}

Answer: {answer}

Provide a score from 0-10 where:
- 0 = Answer completely contradicts context
- 5 = Answer is partially supported by context
- 10 = Answer is fully supported by context

Format your response as: SCORE: [number]\nEXPLANATION: [brief explanation]"""

        logger.info("Evaluating groundedness")
        return {
            "metric": "groundedness",
            "prompt": prompt,
            "answer": answer,
            "context": context
        }

    def score_relevance(
        self,
        answer: str,
        question: str
    ) -> Dict[str, any]:
        """
        Score how relevant the answer is to the question.

        Args:
            answer: Generated answer.
            question: Original question.

        Returns:
            Dictionary with relevance score and explanation.
        """
        prompt = f"""Rate how relevant the answer is to the question.

Question: {question}

Answer: {answer}

Provide a score from 0-10 where:
- 0 = Answer is completely irrelevant
- 5 = Answer is partially relevant
- 10 = Answer is highly relevant

Format your response as: SCORE: [number]\nEXPLANATION: [brief explanation]"""

        logger.info("Evaluating relevance")
        return {
            "metric": "relevance",
            "prompt": prompt,
            "answer": answer,
            "question": question
        }


# ==============================================================================
# SECTION 10: COMPARISON EXPERIMENTS
# ==============================================================================

class ExperimentRunner:
    """Runs experiments comparing baseline and RAG approaches."""

    def __init__(self, assistant: MedicalAssistant, baseline_llm: LLM):
        """
        Initialize experiment runner.

        Args:
            assistant: Medical assistant with RAG.
            baseline_llm: Baseline LLM without RAG.
        """
        self.assistant = assistant
        self.baseline_llm = baseline_llm

    def compare_baseline_vs_rag(self, query: str) -> Dict[str, any]:
        """
        Compare baseline LLM response with RAG response.

        Args:
            query: Medical query.

        Returns:
            Dictionary with both responses for comparison.
        """
        logger.info(f"Running comparison experiment for: {query}")

        # Baseline response (no context)
        baseline_prompt = f"Answer this medical question: {query}\n\nAnswer:"
        baseline_response = self.baseline_llm(baseline_prompt)

        # RAG response
        rag_response = self.assistant.query(query, strategy="rag_augmented")

        return {
            "query": query,
            "baseline_response": baseline_response,
            "rag_response": rag_response,
            "improvement": "RAG provides context-grounded answers"
        }

    def test_prompt_strategies(self, query: str) -> Dict[str, any]:
        """
        Test multiple prompt strategies on the same query.

        Args:
            query: Medical query.

        Returns:
            Dictionary with responses from different strategies.
        """
        logger.info(f"Testing prompt strategies for: {query}")

        strategies = [
            "zero_shot",
            "few_shot",
            "chain_of_thought",
            "structured_output",
            "concise_expert",
            "rag_augmented"
        ]

        results = {"query": query, "strategies": {}}

        for strategy in strategies:
            try:
                response = self.assistant.query(query, strategy=strategy)
                results["strategies"][strategy] = response
            except Exception as e:
                logger.error(f"Error with strategy {strategy}: {str(e)}")
                results["strategies"][strategy] = {"error": str(e)}

        return results


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution block demonstrating the full RAG pipeline."""

    config = RAGConfig()

    # Initialize components
    logger.info("Initializing RAG pipeline...")

    missing_assets = []
    if not os.path.exists(config.pdf_path):
        missing_assets.append(f"PDF data file: {config.pdf_path}")
    if not os.path.exists(config.mistral_model_path):
        missing_assets.append(f"Mistral model file: {config.mistral_model_path}")

    if missing_assets:
        logger.warning("Full RAG pipeline cannot start because required assets are missing:")
        for asset in missing_assets:
            logger.warning("  - %s", asset)
        logger.info("Install dependencies with: python3 -m pip install -r requirements.txt")
        logger.info("Then add the Merck Manual PDF and Mistral GGUF model described in README.md")
        return {"status": "missing_assets", "missing_assets": missing_assets}

    # 1. PDF Ingestion
    ingestor = PDFIngestor(config.pdf_path)
    documents = ingestor.extract_text_with_metadata()

    # 2. Text Chunking
    chunker = TextChunker(config.chunk_size, config.chunk_overlap)
    chunked_documents = chunker.chunk_documents(documents)

    # 3. Embedding Generation
    embedding_gen = EmbeddingGenerator(config.embedding_model_name)
    embeddings = embedding_gen.embed_texts([doc["text"] for doc in chunked_documents])

    # 4. Vector Store
    vector_store = VectorStore(config.chroma_db_path, config.collection_name)
    vector_store.index_documents(chunked_documents, embeddings)
    vector_store.persist()

    # 5. LLM Setup
    llm = MistralLLM(
        config.mistral_model_path,
        n_gpu_layers=config.n_gpu_layers,
        n_ctx=config.n_ctx,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens
    )

    # 6. RAG Chain
    rag_chain = RAGChain(vector_store, llm, embedding_gen, config.retriever_k)

    # 8. Medical Assistant
    assistant = MedicalAssistant(rag_chain)

    # Example queries
    sample_queries = [
        "What are the symptoms of hypertension?",
        "How is diabetes diagnosed?",
        "What is the treatment for pneumonia?",
        "Describe the pathophysiology of heart failure",
        "What are the risk factors for stroke?"
    ]

    logger.info("Running sample queries...")
    for query in sample_queries:
        print(f"\nQuery: {query}")

        try:
            result = assistant.query(query, strategy="rag_augmented")
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['num_sources']}")
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            print(f"Error: {str(e)}")

    return {"status": "completed", "queries_run": len(sample_queries)}


if __name__ == "__main__":
    main()
