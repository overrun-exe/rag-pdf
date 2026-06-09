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
    k: int = 3

@app.post("/ask")
def ask(q: Query):
    docs_and_scores = db.similarity_search_with_score(
        q.question,
        k=q.k
    )

    results = []

    context_parts = []

    for doc, score in docs_and_scores:
        results.append({
            "text": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score)
        })

        context_parts.append(doc.page_content)

    context = "\n\n---\n\n".join(context_parts)

    return {
        "question": q.question,
        "context": context,
        "results": results
    }