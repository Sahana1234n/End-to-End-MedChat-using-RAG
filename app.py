import gradio as gr
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Initialize embeddings and vector store
embeddings = download_embeddings()
index_name = "medical-bot"
vector_store = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize model
model = ChatGroq(model="llama-3.1-8b-instant")

# Prompt setup
prompt = ChatPromptTemplate(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Function to handle user input
def chat(msg):
    response = rag_chain.invoke({"input": msg})
    return response['answer']

# Gradio Interface
iface = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(lines=2, placeholder="Type your question here..."),
    outputs="text",
    title="MedChat",
    description="Ask medical-related questions and get AI-powered responses."
)

if __name__ == "__main__":
    iface.launch()
