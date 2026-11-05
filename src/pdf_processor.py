import os
import tempfile
from typing import Tuple
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

def build_pdf_vectorstore(
    file_bytes: bytes,
    api_key: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Tuple[object, int, int]:
    """
    Bouwt een Chroma vectorstore op uit PDF-bytes met een stabiel Google embedding-model.
    """
    # Zorg dat de API-sleutel correct in de omgeving staat voor onderliggende componenten
    os.environ["GOOGLE_API_KEY"] = api_key
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        # 1. Documenten inladen via PyMuPDF
        loader = PyMuPDFLoader(temp_path)
        documents = loader.load()

        # 2. Tekst opsplitsen in chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        docs = text_splitter.split_documents(documents)

        # 3. Initialiseer het structurele Google Embedding model
        # Er is expliciet gekozen voor gemini-embedding-001 met task_type 'retrieval_document'
        # Dit zorgt voor de meest stabiele en nauwkeurige verwerking binnen een RAG-systeem.
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
            task_type="retrieval_document"
        )

        # 4. Vectorstore opbouwen
        vectorstore = Chroma.from_documents(docs, embeddings)
        return vectorstore, len(documents), len(docs)

    except Exception as e:
        raise RuntimeError(f"Error processing PDF: {e}")
    finally:
        # Altijd netjes het tijdelijke bestand opruimen
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
