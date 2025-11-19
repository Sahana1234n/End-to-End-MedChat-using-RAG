from langchain.document_loaders import PyPDFLoader , DirectoryLoader
from typing import List
from langchain.text_splitter  import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

def load_pdf(data):
    loader = DirectoryLoader(
        data ,
        glob= "*.pdf" ,
        loader_cls=PyPDFLoader 
    )

    documents = loader.load()
    return documents




def filter_to_minimal_docs(docs:List[Document])->List[Document]:
    """Given a list of documents objects, return a new list of document objects
    which containg source in metadata and the original page_content"""

    minimal_docs:List[Document] = []

    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content , 
                metadata = {"source":src}

            )
        )

    return minimal_docs


def text_splitter(docs:List[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500 , 
        chunk_overlap = 50 
    )
    chunks = splitter.split_documents(docs)

    return chunks


def download_embeddings():
    """Download and return the embedding model """
    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2" , 
    
    )

    return embeddings