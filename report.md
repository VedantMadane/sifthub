Embedding Model Evaluation Approach:

The approach evaluates multiple embedding models for retrieval accuracy on enterprise documentation. It is targeted at RAG (Retrieval-Augmented Generation) systems to measure how well embedding models can retrieve relevant document chunks given a query.

The three embedding models evaluated are:
- all-mpnet-base-v2 (general purpose from SentenceTransformers)
- bge-large-en-v1.5 (BGE model from BAAI optimized for retrieval tasks)
- e5-large-v2 (Microsoft's E5 semantic search model)

Evaluation is performed on documentation chunks extracted from the ACME Enterprise Platform, with ground truth relevance scores (0-1) for chunks per question.

Metrics calculated per question include:
- Precision@K
- Recall@K
- F1 Score
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)

The script performs the following steps:
1. Load and encode document chunks using each embedding model.
2. Compute cosine similarity between query and document chunk embeddings.
3. Rank document chunks by similarity scores.
4. Evaluate ranked results against ground truth relevance using the metrics above.
5. Generate plots and save report JSON summarizing results per question and model.

Summary:
- The evaluation helps determine the best embedding model for enterprise retrieval tasks.
- Metrics cover relevance, ranking quality, and coverage capabilities.
- Visualization aids in comparing models on key retrieval performance aspects.

  
