from typing import Dict, List

from pymilvus import Collection, connections


class ZillizVectorSearch:
    def __init__(
        self,
        zilliz_user: str,
        zilliz_password: str,
        zilliz_uri: str,
        collection_name: str,
        output_fields: List[str],
        sparse_embeddings,
        dense_embeddings,
        colbert_reranker
    ) -> None:
        """
        Initializes the VectorSearch class with necessary details.
        """
        self.zilliz_user = zilliz_user
        self.zilliz_password = zilliz_password
        self.zilliz_uri = zilliz_uri
        self.collection_name = collection_name
        self.output_fields = output_fields
        self.sparse_embeddings = sparse_embeddings
        self.dense_embeddings = dense_embeddings
        self.reranker = colbert_reranker

        # Connect to the collection
        self.collection = self.retrieve_collection()

    def retrieve_collection(self) -> Collection:
        """
        Retrieve the collection from Zilliz.
        """
        connections.connect(
            uri=self.zilliz_uri, user=self.zilliz_user, password=self.zilliz_password
        )
        return Collection(self.collection_name)

    def search(self, query: str, embedding_type: str, top_k: int) -> List[Dict]:
        """
        Perform the search on a specific vector type (dense or sparse).
        """
        if embedding_type == "dense":
            anns_field = "dense_vector"
            metric_type = "COSINE"
            query = self.dense_embeddings.encode_queries([query])["dense"]
        elif embedding_type == "sparse":
            anns_field = "sparse_vector"
            metric_type = "IP"
            query = self.sparse_embeddings.encode_queries([query])
        else:
            raise ValueError("Invalid embedding type. Must be either 'dense' or 'sparse'.")

        results = self.collection.search(
            query,
            anns_field=anns_field,
            limit=top_k,
            param={"metric_type": metric_type},
            output_fields=self.output_fields,
        )
        results = [result.to_dict()["entity"] for result in results[0]]

        return results

    def hybrid_search(self, query: str, top_k_each: int = 5, top_k_final: int = 5) -> List[Dict]:
        """
        Perform a hybrid search combining dense and sparse searches.
        """
        pk_set = set()
        unique_results = []

        # Search for both dense and sparse results
        dense_results = self.search(query, "dense", top_k_each)
        sparse_results = self.search(query, "sparse", top_k_each)
        all_results = dense_results + sparse_results

        # Combine dense and sparse results and remove duplicates
        for result in all_results:
            if result["pk"] not in pk_set:
                pk_set.add(result["pk"])
                unique_results.append(result)

        # Rerank the results
        sorted_indices = self.reranker.rerank_docs_query(
            [result["text"] for result in unique_results], query, top_k=top_k_final, return_indices=True
        )
        ranked_results = [unique_results[i] for i in sorted_indices]
        return ranked_results
    