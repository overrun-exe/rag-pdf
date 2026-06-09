from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Query):
    docs = db.similarity_search(q.question, k=3)

    return {
        "question": q.question,
        "chunks": [
            {
                "text": d.page_content,
                "metadata": d.metadata
            }
            for d in docs
        ]
    }