from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = Ollama(model="llama3")

while True:
    question = input("\nQuestion: ")

    docs = db.similarity_search(question, k=3)

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Ты помощник, отвечай только по контексту.

Контекст:
{context}

Вопрос:
{question}
"""

    answer = llm.invoke(prompt)

    print("\nANSWER:\n", answer)