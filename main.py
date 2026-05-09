from sentence_transformers import SentenceTransformer
from data.manuals import documents


def generate_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(documents)

    print("Quantidade de documentos:", len(documents))
    print("Dimensão dos embeddings:", len(embeddings[0]))

    return embeddings


if __name__ == "__main__":
    generate_embeddings()