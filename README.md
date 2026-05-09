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

## HNSW: M e ef_construction

O algoritmo HNSW utiliza uma estrutura hierárquica de grafos para acelerar buscas vetoriais aproximadas.

- M:
Controla a quantidade máxima de conexões entre nós do grafo. Valores maiores aumentam precisão, porém consomem mais memória RAM.

- ef_construction:
Define o esforço computacional durante a construção do índice. Valores maiores tornam a indexação mais lenta, mas aumentam a qualidade das buscas.

Comparado ao KNN exato, o HNSW reduz drasticamente o custo computacional de busca, sacrificando uma pequena parcela da precisão em troca de velocidade e escalabilidade.