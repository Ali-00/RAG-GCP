from pathlib import Path
from typing import Union

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.storage import VECTORSTORE_DIR


def ingest_pdf(file_path: Union[str, Path]) -> None:
    """
    Ingest a PDF file into a FAISS vector store.
    - Loads PDF
    - Splits into chunks
    - Generates embeddings
    - Creates or updates FAISS index
    - Persists to disk
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = PyPDFLoader(file_path.as_posix())
    documents = loader.load()

    if not documents:
        raise ValueError("No content found in PDF")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Text splitting resulted in zero chunks")

    embeddings = OpenAIEmbeddings()

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    if any(VECTORSTORE_DIR.iterdir()):
        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR.as_posix(),
            embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(VECTORSTORE_DIR.as_posix())
