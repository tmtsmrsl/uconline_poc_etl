from typing import List

import numpy as np
from fastembed import LateInteractionTextEmbedding


class ColBERTReranker(LateInteractionTextEmbedding):
    def __init__(self, model_name: str = "answerdotai/answerai-colbert-small-v1", **kwargs) -> None:
        """
        Initializes the ColBERTReranker with the given model name.
        """
        super().__init__(model_name=model_name, **kwargs)

    def compute_relevance_scores(self, query_embedding: np.array, document_embeddings: np.array, k: int) -> List[int]:
        """
        Compute relevance scores for top-k documents given a query.
        """
        # Compute batch dot-product of query_embedding and document_embeddings
        # Resulting shape: [num_documents, num_query_terms, max_doc_length]
        scores = np.matmul(query_embedding, document_embeddings.transpose(0, 2, 1))

        # Apply max-pooling across document terms (axis=2) to find the max similarity per query term
        # Shape after max-pool: [num_documents, num_query_terms]
        max_scores_per_query_term = np.max(scores, axis=2)

        # Sum the scores across query terms to get the total score for each document
        # Shape after sum: [num_documents]
        total_scores = np.sum(max_scores_per_query_term, axis=1)

        # Sort the documents based on their total scores and get the indices of the top-k documents
        sorted_indices = np.argsort(total_scores)[::-1][:k]

        return list(sorted_indices)

    def rerank_docs_query(self, docs: List[str], query: str, top_k: int = 5, return_indices: bool = False) -> List[str] or List[int]:
        """
        Rerank the documents based on the query. Can return the indices of the top-k documents or the documents themselves.
        """
        document_embeddings = list(self.embed(docs))
        query_embeddings = list(self.embed([query]))
        
        sorted_indices = self.compute_relevance_scores(
        np.array(query_embeddings[0]), np.array(document_embeddings), k=top_k
        )
        
        if return_indices:
            return sorted_indices
        else:
            return [docs[i] for i in sorted_indices]
        