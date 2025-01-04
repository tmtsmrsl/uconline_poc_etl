import re

import numpy as np
from fastembed import LateInteractionTextEmbedding
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from pymilvus import Collection, connections
from typing_extensions import Dict, List, TypedDict


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
        
class ZillizVectorSearch:
    def __init__(
        self,
        zilliz_user: str,
        zilliz_password: str,
        zilliz_uri: str,
        collection_name: str,
        sparse_embeddings,
        dense_embeddings,
        colbert_reranker,
    ) -> None:
        """
        Initializes the VectorSearch class with necessary details.
        """
        self.zilliz_user = zilliz_user
        self.zilliz_password = zilliz_password
        self.zilliz_uri = zilliz_uri
        self.collection_name = collection_name
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

        output_fields = [
            "pk",
            "start_index",
            "data_block_ranges",
            "module_title",
            "subsection",
            "submodule_title",
            "submodule_url",
            "text",
        ]

        results = self.collection.search(
            query,
            anns_field=anns_field,
            limit=top_k,
            param={"metric_type": metric_type},
            output_fields=output_fields,
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
    
class State(TypedDict):
    question: str
    sources: List[Dict]
    formatted_sources: Dict
    answer: str

class PromptManager:
    @staticmethod
    def load_generate_prompt(course_name) -> ChatPromptTemplate:
        generate_system_prompt = f"""You're a helpful personalized tutor for {course_name}. Given a user question and some course contents, answer the question based on the course contents and justify your answer by providing an accurate inline citation of the source IDs. If none of the course content answer the question, just say: "Sorry, I can't find any relevant course content related to your question". 
Follow the following format STRICTLY for the final answer:
This is an example of inline citation^[5]. One sentence can have multiple inline citations^[3], and the inline citation can also consist of multiple numbers^[7]^[8]. 

Here are the course contents (not visible to the user):
{{sources}}"""

        return ChatPromptTemplate.from_messages([
            ("system", generate_system_prompt),
            ("human", "{question}"),
        ])

class SourceFormatter:
    @staticmethod
    def _merge_overlapping_sources(sources: List[Dict]) -> List[Dict]:
        submodule_url_dict = {}

        # Group sources by submodule_url
        for source in sources:
            source['end_index'] = source['start_index'] + len(source['text'])
            if source['submodule_url'] not in submodule_url_dict:
                submodule_url_dict[source['submodule_url']] = []
            submodule_url_dict[source['submodule_url']].append(source)

        # Merge overlapping sources for each submodule_url
        for submodule_url, sources in submodule_url_dict.items():
            if len(sources) > 1:
                new_sources = []
                # Sort documents by start_index
                sorted_sources = sorted(sources, key=lambda x: x['start_index'])
                for i, source in enumerate(sorted_sources):
                    if i == 0:
                        new_sources.append(source)
                    else:
                        # Check if the current source overlaps with the last source in new_sources
                        if source['start_index'] < new_sources[-1]['end_index']:
                            # note that we don't use the end_index of the last source directly as the indices are relative to the original submodule content
                            non_overlapping_text = source['text'][new_sources[-1]['end_index'] - source['start_index']:]
                            new_sources[-1]['text'] += non_overlapping_text
                            new_sources[-1]['end_index'] = source['end_index']
                        else:
                            new_sources.append(source)
                # Update the submodule_url_dict with the merged sources
                submodule_url_dict[submodule_url] = new_sources
                
        # Flatten the dictionary back to a list
        merged_sources = []
        for sources in submodule_url_dict.values():
            merged_sources.extend(sources)
            
        return merged_sources

    @staticmethod
    def _split_source_by_block(source: Dict) -> List[Dict]:
        source_splits = []
        start_index = source['start_index']
        end_index = start_index + len(source['text'])
        contextual_header = f"Below are content snippet of: {source['module_title']} - {source['subsection']}: {source['submodule_title']}"
                
        for block_id, block_range in source['data_block_ranges'].items():
            if start_index <= block_range['char_end'] and end_index >= block_range['char_start']:
                adjust_start = max(0, block_range['char_start']-start_index)
                adjusted_end = min(len(source['text']), block_range['char_end']-start_index)
                text = source['text'][adjust_start:adjusted_end]
                source_splits.append(
                    {   
                        "block_id": block_id,
                        "text": text,
                    }
                )
        
        source_dict = {
            "submodule_url": source['submodule_url'],
            "submodule_title": source['submodule_title'],
            "contextual_header": contextual_header,
            "source_splits": source_splits
        }
        
        return source_dict

    def format_sources_for_llm(self, sources: List[Dict]) -> Dict:
        merged_sources = self._merge_overlapping_sources(sources)
        source_dicts = []
        source_id = 0
        final_formatted_sources = ""
        
        for source in merged_sources:
            source_dict = self._split_source_by_block(source)
            formatted_splits = ""
            for split in source_dict['source_splits']:
                split['source_id'] = source_id
                formatted_splits += f"[{split['source_id']}]\n{split['text'].strip()}\n"
                source_id += 1
            source_dict['source_ids'] = [split['source_id'] for split in source_dict['source_splits']]
            source_dicts.append(source_dict)
            final_formatted_sources += source_dict['contextual_header'] + ":\n" + formatted_splits + "===\n\n"
            
        return {"content": final_formatted_sources, "source_dicts": source_dicts}

class CitationFormatter:
    @staticmethod
    def _deduplicate_consecutive_citations(text: str) -> str:
    # Find groups of consecutive citations and deduplicate them
        def replace_consecutive(match):
            citations = re.findall(r'\[(\d+)\]', match.group(0))
            unique_citations = []
            for citation in citations:
                # Only remove duplicates within this consecutive group
                if citation not in unique_citations:
                    unique_citations.append(citation)
            return ''.join(f'[{citation}]' for citation in unique_citations)
        
        # Pattern to match one or more consecutive citations
        pattern = r'(?:\[\d+\])+' 
        return re.sub(pattern, replace_consecutive, text)
    
    def format_final_answer(self, result: Dict) -> Dict:
        # extract the citation ids from the answer
        citation_pattern = re.compile(r"\^\[(\d+)\]")
        citation_ids = citation_pattern.findall(result['answer'])
        citation_ids = set([int(id) for id in citation_ids])

        # find the source for each citation
        citation_data = []
        source_metadata = result['formatted_sources']['source_dicts']

        for submodule_source in source_metadata:
            matching_ids = set(submodule_source['source_ids']).intersection(citation_ids)
            if matching_ids:
                citation = {"submodule_url": submodule_source['submodule_url'], 
                            "submodule_title": submodule_source['submodule_title'],
                            "old_citation_ids": [], 
                            "block_ids": []}
                for source_split in submodule_source['source_splits']:
                    if source_split['source_id'] in matching_ids:
                        citation['old_citation_ids'].append(source_split['source_id'])
                        citation['block_ids'].append(source_split['block_id'])
                citation_data.append(citation)

        # Sort by the minimum value in 'citation_ids'
        citation_data = sorted(citation_data, key=lambda x: min(x['old_citation_ids']))

        # process the final citation url and id
        new_citation_id = 1
        for citation in citation_data:
            citation['final_url'] = citation['submodule_url'] + "/block/" + ",".join(citation['block_ids'])
            citation['new_citation_id'] = new_citation_id
            new_citation_id += 1

        # reformat the answer with the new citation ids
        final_answer = result['answer']
        for citation in citation_data:
            for old_citation_id in citation['old_citation_ids']:
                final_answer = final_answer.replace(f"^[{old_citation_id}]", f"[{citation['new_citation_id']}]")
                
        final_citation = {}
        for citation in citation_data:
            final_citation[citation['new_citation_id']] = {"url": citation['final_url'], "title": citation['submodule_title']}
        
        # deduplicate the final citation
        final_answer = self._deduplicate_consecutive_citations(final_answer)
        
        return {"content": final_answer, "citation": final_citation}
    
class QAPipeline():
    def __init__(self, llm, vector_search: ZillizVectorSearch, course_name: str):
        self.llm = llm
        self.vector_search = vector_search
        self.prompt_manager = PromptManager()
        self.source_formatter = SourceFormatter()
        self.citation_formatter = CitationFormatter()
        self.generate_prompt = self.prompt_manager.load_generate_prompt(course_name)
        self.graph = self.build_graph()
    
    def retrieve(self, state: State):
        retrieved_sources = self.vector_search.hybrid_search(query=state["question"], top_k_final=4)
        formatted_sources = self.source_formatter.format_sources_for_llm(retrieved_sources)
        return {"sources": retrieved_sources, "formatted_sources": formatted_sources}

    def generate(self, state: State):
        messages = self.generate_prompt.invoke({"question": state["question"], "sources": state["formatted_sources"]["content"]})
        response = self.llm.invoke(messages)
        return {"answer": response.content}
    
    def build_graph(self):
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        return graph_builder.compile()
    
    def run(self, query: str) -> Dict:
        result = self.graph.invoke({"question": query})
        final_answer = self.citation_formatter.format_final_answer(result)
        return final_answer
        