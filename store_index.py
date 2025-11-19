import os
from src.helper import load_pdf , filter_to_minimal_docs , text_splitter , download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore



from dotenv import load_dotenv
load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] =GROQ_API_KEY

extracted_data = load_pdf(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_splitter(filter_data)

embedding = download_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key = pinecone_api_key)

index_name = "medical-bot"

if not pc.has_index(index_name):
    pc.create_index(
        name= index_name , 
        metric= "cosine",
        dimension=384,
        spec=ServerlessSpec(cloud="aws" , region="us-east-1")
    )

index = pc.Index(index_name)   

vector_store = PineconeVectorStore.from_documents( 
    documents = text_chunks,
    embedding= embedding , 
    index_name= index_name
)

