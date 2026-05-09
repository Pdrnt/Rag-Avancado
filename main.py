import chromadb

from sentence_transformers import SentenceTransformer
from data.manuals import documents


model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()

collection = client.create_collection(
    name="medical_manuals"
)


def generate_embeddings():
    return model.encode(documents).tolist()


def index_documents():
    embeddings = generate_embeddings()

    ids = [f"doc_{i}" for i in range(len(documents))]

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids
    )

    print("Documentos indexados com sucesso no HNSW.")


if __name__ == "__main__":
    index_documents()