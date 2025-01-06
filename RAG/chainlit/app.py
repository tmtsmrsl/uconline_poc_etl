import os
from typing import Dict, List

import chainlit as cl
import joblib
import torch
from chainlit.input_widget import Select, Slider, Switch
from dotenv import load_dotenv
from langchain_cerebras import ChatCerebras
from langchain_openai import ChatOpenAI
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from RAG.utils import config
from RAG.utils.ColBERTReranker import ColBERTReranker
from RAG.utils.QAPipeline import QAPipeline
from RAG.utils.ZillizVectorSearch import ZillizVectorSearch


def load_env_vars() -> Dict:
    """Load environment variables from the .env file."""
    load_dotenv()
    session_env = {
        "ZILLIZ_URI": os.getenv("ZILLIZ_URI"),
        "ZILLIZ_USER": os.getenv("ZILLIZ_USER"),
        "ZILLIZ_PASSWORD": os.getenv("ZILLIZ_PASSWORD"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "CEREBRAS_API_KEY": os.getenv("CEREBRAS_API_KEY"),
        "LANGCHAIN_API_KEY": os.getenv("LANGCHAIN_API_KEY")
    }
    cl.user_session.set("session_env", session_env)
    
def load_config() -> Dict:
    """Load configuration settings for the app."""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    session_config = {
        "DEVICE": DEVICE, 
        "SPARSE_EMBEDDINGS_PATH": config.SPARSE_EMBEDDINGS_PATH,
        "COLBERT_MODEL_NAME": config.COLBERT_MODEL_NAME,
        "ZILLIZ_COLLECTION_NAME": config.ZILLIZ_COLLECTION_NAME,
        "COURSE_NAME": config.COURSE_NAME,
        "LLM_TEMPERATURE": config.LLM_TEMPERATURE,
        "LLM_MAX_RETRIES": config.LLM_MAX_RETRIES
    }
    cl.user_session.set("session_config", session_config)
    
def initialize_vector_search() -> ZillizVectorSearch:
    """Initialize vector search with Zilliz."""
    session_env = cl.user_session.get("session_env")
    session_config = cl.user_session.get("session_config")
    
    dense_embeddings = BGEM3EmbeddingFunction(use_fp16=False, device=session_config['DEVICE'], return_dense=True, return_sparse=False)
    sparse_embeddings = joblib.load(session_config['SPARSE_EMBEDDINGS_PATH'])
    colbert_reranker = ColBERTReranker(model_name=session_config['COLBERT_MODEL_NAME'])
    
    vector_search = ZillizVectorSearch(session_env["ZILLIZ_USER"], session_env["ZILLIZ_PASSWORD"], session_env["ZILLIZ_URI"], 
                            session_config['ZILLIZ_COLLECTION_NAME'], sparse_embeddings, dense_embeddings, colbert_reranker)
    
    cl.user_session.set("vector_search", vector_search)

async def send_initial_message():
    """Send an initial welcome message to the user."""
    session_config = cl.user_session.get("session_config")
    message_content = (
        f"Hi, I am UCOnline AI Assistant which can help you learn about {session_config['COURSE_NAME']}. " 
        "Ask me anything related to the course, and I will try to answer based on the course content!"
    )
    msg = cl.Message(content=message_content)
    await msg.send()

def format_citation_elements(citations: Dict) -> List:
    if citations:
        formatted_citations = ""
        for id, source in citations.items():
            formatted_citations += f"{id}. [{source['title']}]({source['url']})\n"
        elements = [
            cl.Text(name="Sources", content=formatted_citations, display="inline")
            ]
    else:
        elements = []
    return elements

@cl.on_settings_update
async def setup_pipeline(settings):
    """Setup the QAPipeline with the selected model."""
    session_env = cl.user_session.get("session_env")
    session_config = cl.user_session.get("session_config")
    vector_search = cl.user_session.get("vector_search")

    selected_model = settings["Model"]

    if selected_model in ["gpt-4o", "gpt-4o-mini"]:
        llm = ChatOpenAI(api_key=session_env["OPENAI_API_KEY"], model=selected_model, temperature=session_config['LLM_TEMPERATURE'], max_retries=session_config['LLM_MAX_RETRIES'])
    elif selected_model in ["llama-3.3-70b"]:
        llm = ChatCerebras(api_key=session_env["CEREBRAS_API_KEY"], model=selected_model, temperature=session_config['LLM_TEMPERATURE'], max_retries=session_config['LLM_MAX_RETRIES'])
        
    qa_pipeline = QAPipeline(llm, vector_search, course_name=session_config['COURSE_NAME'])
    cl.user_session.set("qa_pipeline", qa_pipeline)


@cl.on_chat_start
async def start():
    # Enable the tracing feature of Langsmith
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    
    # Load environment variables and configuration settings
    load_env_vars()
    load_config()
    
    
    # intialize vector search
    initialize_vector_search()
    
    # Setup the settings interface and QAPipeline with the default model
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Model",
                values=["gpt-4o", "gpt-4o-mini", "llama-3.3-70b"],
                initial_index=0,
            )
        ]
    ).send()
    await setup_pipeline(settings)
    
    await send_initial_message()

@cl.on_message
async def main(message: cl.Message):
    qa_pipeline = cl.user_session.get("qa_pipeline") 

    response = qa_pipeline.run(query=message.content)
    answer = response["content"]
    citations = response["citation"]
    citation_elements = format_citation_elements(citations)

    await cl.Message(content=answer, elements=citation_elements).send() 

