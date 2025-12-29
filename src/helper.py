from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List

# Load PDFs from a directory
def load_pdf_files(data_path: str) -> List[Document]:
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()

# Filter relevant metadata
def filter_relevant_info(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(page_content=doc.page_content, metadata={"source": src})
        )
    return minimal_docs

# Split documents into smaller chunks
def text_split(minimal_docs: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    return text_splitter.split_documents(minimal_docs)

# API-based embeddings (no local model load)
def download_embedding() -> HuggingFaceEmbeddings:
    """
    Returns HuggingFace API-based embeddings.
    Memory-friendly, uses Hugging Face hosted inference.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        # Token is read from HUGGINGFACEHUB_API_TOKEN environment variable
    )
