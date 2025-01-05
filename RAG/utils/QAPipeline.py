from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import Dict, List, TypedDict
from RAG.utils.CitationFormatter import CitationFormatter
from RAG.utils.SourceFormatter import SourceFormatter
from RAG.utils.ZillizVectorSearch import ZillizVectorSearch


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
        