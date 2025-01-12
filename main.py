from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import cassio
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables
load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the connection to Astra DB
cassio.init(
    token=ASTRA_DB_APPLICATION_TOKEN,
    database_id=ASTRA_DB_ID,
)

# Setup LangChain embeddings and vector store
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="ai_demo_drinks",
    session=None,
    keyspace=None,
)

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# FastAPI setup
app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to specific frontend URLs for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Mocktail Generator Backend is Running"}

@app.post("/ask")
async def ask_question(question: str):
    try:
        # Query the vector index and retrieve an answer
        answer = astra_vector_index.query(question, llm=llm).strip()

        # Get similar documents for context
        documents = astra_vector_store.similarity_search_with_score(question, k=3)
        similar_docs = [{"score": score, "content": doc.page_content} for doc, score in documents]

        return {"answer": answer, "similar_documents": similar_docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
