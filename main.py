import chromadb

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from data.manuals import documents


model = SentenceTransformer("all-MiniLM-L6-v2")

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

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


def hyde_transform(query):
    hypothetical_document = f"""
    Paciente apresenta sintomas clínicos compatíveis com:
    {query}

    Possível presença de quadro neurológico associado à cefaleia pulsátil,
    fotofobia, náusea e sensibilidade à luz.
    """

    print("\nDocumento Hipotético Gerado (HyDE):\n")
    print(hypothetical_document)

    embedding = model.encode(hypothetical_document).tolist()

    return embedding


def retrieve_documents(query_embedding):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )

    print("\nTOP 10 DOCUMENTOS RECUPERADOS:\n")

    for i, doc in enumerate(results["documents"][0], start=1):
        print(f"{i}. {doc}")

    return results["documents"][0]


def rerank_documents(query, retrieved_docs):
    pairs = [[query, doc] for doc in retrieved_docs]

    scores = cross_encoder.predict(pairs)

    ranked_results = sorted(
        zip(retrieved_docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTOP 3 DOCUMENTOS APÓS RE-RANKING:\n")

    for i, (doc, score) in enumerate(ranked_results[:3], start=1):
        print(f"{i}. Score: {score:.4f}")
        print(doc)
        print()


if __name__ == "__main__":
    index_documents()

    query = "dor de cabeça latejante e luz incomodando"

    hyde_embedding = hyde_transform(query)

    retrieved_docs = retrieve_documents(hyde_embedding)

    rerank_documents(query, retrieved_docs)