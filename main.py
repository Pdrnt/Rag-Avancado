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


if __name__ == "__main__":
    index_documents()

    query = "dor de cabeça latejante e luz incomodando"

    hyde_embedding = hyde_transform(query)