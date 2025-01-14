import os
from typing import Dict

import joblib
import torch
from dotenv import load_dotenv
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from RAG.utils import config
from RAG.utils.ColBERTReranker import ColBERTReranker
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
    return session_env

    
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
        "LLM_MAX_RETRIES": config.LLM_MAX_RETRIES,
        "ZILLIZ_OUTPUT_FIELDS": config.ZILLIZ_OUTPUT_FIELDS
    }
    return session_config

def initialize_vector_search(session_env, session_config) -> ZillizVectorSearch:
    """Initialize vector search with Zilliz."""
    dense_embeddings = BGEM3EmbeddingFunction(use_fp16=False, device=session_config['DEVICE'], return_dense=True, return_sparse=False)
    sparse_embeddings = joblib.load(session_config['SPARSE_EMBEDDINGS_PATH'])
    colbert_reranker = ColBERTReranker(model_name=session_config['COLBERT_MODEL_NAME'])
    
    return ZillizVectorSearch(session_env["ZILLIZ_USER"], session_env["ZILLIZ_PASSWORD"], session_env["ZILLIZ_URI"], 
                            session_config['ZILLIZ_COLLECTION_NAME'], session_config['ZILLIZ_OUTPUT_FIELDS'],
                            sparse_embeddings, dense_embeddings, colbert_reranker)