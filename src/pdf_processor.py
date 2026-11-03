import os
import tempfile
from typing import Tuple
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

#Create a temporary file to store the PDF bytes
def build_pdf_vectorstore(file_bytes: bytes, 
                          api_key: str,
                          chunk_size: int = 1000, 
                          chunk_overlap: int = 200) -> Tuple[FAISS, int, int]:
    """
    Builds a FAISS vector store from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.
        chunk_size (int): The size of each text chunk. Default is 1000.
        chunk_overlap (int): The overlap between chunks. Default is 200.

    Returns:
        tuple: A tuple containing the FAISS vector store and the list of text chunks.
    """

    #Set the API key for Google Generative AI
    os.environ["GOOGLE_API_KEY"] = api_key

    #Load the PDF file using PyMuPDFLoader in temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name 

    try:
        #Load the PDF file using PyMuPDFLoader
        loader = PyMuPDFLoader(temp_path)
        documents = loader.load()

        #Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        #Split the documents into chunks
        docs = text_splitter.split_documents(documents)

        #Create embeddings using Google Generative AI for semantic search
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",
            google_api_key=api_key)

        #Create a FAISS vector store from the texts and embeddings
        vectorstore = FAISS.from_documents(docs, embeddings)

        return vectorstore, len(documents), len(docs)


    except Exception as e:
        raise RuntimeError(f"Error processing PDF: {e}")
        return None, 0, 0
    

    #Clean up the temporary file in the finally block 
    # to ensure it is always removed

    finally:
        #Clean up the temporary file
        os.remove(temp_path)

     
