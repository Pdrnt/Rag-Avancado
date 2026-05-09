Laboratório 09 - Arquitetura RAG Avançada

## Objetivo
Implementar um pipeline de Retrieval-Augmented Generation (RAG) utilizando:

- HNSW
- HyDE
- Cross-Encoder
- Busca Vetorial
- Re-ranking semântico

## Tecnologias Utilizadas

- Python
- ChromaDB
- Sentence Transformers
- OpenAI Embeddings
- FAISS/HNSW

## Estrutura do Projeto

```bash
.
├── main.py
├── requirements.txt
├── README.md
└── data/
Etapas do Pipeline
Construção do índice HNSW
Geração de embeddings
Query Transformation com HyDE
Recuperação vetorial Top-K
Re-ranking com Cross-Encoder
Observações

Este projeto foi desenvolvido para a disciplina de IA Avançada.

Declaração de Uso de IA

"Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por Pedro Lima"