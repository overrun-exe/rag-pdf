from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

while True:
    question = input("\nQuestion: ")

    docs = db.similarity_search(question, k=3)

    print("\n--- TOP CHUNKS ---")
    for i, doc in enumerate(docs):
        print(f"\n[{i+1}]\n{doc.page_content}")