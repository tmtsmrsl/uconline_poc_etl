import json

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import HybridSearch, VectorizedQuery
from langchain_openai import OpenAIEmbeddings


class AzureVectorSearch:
    def __init__(self, search_endpoint: str, search_key: str, index_name: str, embedding_model: OpenAIEmbeddings, output_fields: list[str]):
        self.search_client = SearchClient(endpoint=search_endpoint,
                                        credential=AzureKeyCredential(search_key),
                                        index_name=index_name)
        self.embedding_model = embedding_model
        self.output_fields = output_fields

    def hybrid_search(self, query, top_k_each, top_k_final):
        embedded_query = self.embedding_model.embed_query(query)

        search_results = self.search_client.search(  
            search_text=query,  
            search_fields=["text"],
            hybrid_search=HybridSearch(max_text_recall_size=top_k_each),
            vector_queries= [VectorizedQuery(
                vector=embedded_query, 
                k_nearest_neighbors=top_k_each, 
                fields="dense_vector")],
            top=top_k_final,
            select=self.output_fields,
            query_type="semantic",
            semantic_configuration_name="my-semantic-config"
        )  
        
        results = list(search_results)
        
        for r in results:
            r['index_metadata'] = json.loads(r['index_metadata'])
            
        return results
