import os
from typing import Dict

import joblib
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
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
        # "CEREBRAS_API_KEY": os.getenv("CEREBRAS_API_KEY"),
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
        "LLM_MAX_RETRIES": config.LLM_MAX_RETRIES
    }
    return session_config

def initialize_vector_search(session_env, session_config) -> ZillizVectorSearch:
    """Initialize vector search with Zilliz."""
    dense_embeddings = BGEM3EmbeddingFunction(use_fp16=False, device=session_config['DEVICE'], return_dense=True, return_sparse=False)
    sparse_embeddings = joblib.load(session_config['SPARSE_EMBEDDINGS_PATH'])
    colbert_reranker = ColBERTReranker(model_name=session_config['COLBERT_MODEL_NAME'])
    
    return ZillizVectorSearch(session_env["ZILLIZ_USER"], session_env["ZILLIZ_PASSWORD"], session_env["ZILLIZ_URI"], 
                            session_config['ZILLIZ_COLLECTION_NAME'], sparse_embeddings, dense_embeddings, colbert_reranker)

def setup_pipeline() -> QAPipeline:
    """Setup the QA pipeline with pre-configured settings."""
    session_env = load_env_vars()
    session_config = load_config()

    vector_search = initialize_vector_search(session_env, session_config)
    llm = ChatOpenAI(
        api_key=session_env['OPENAI_API_KEY'],
        model="gpt-4o",
        temperature=session_config['LLM_TEMPERATURE'],
        max_retries=session_config['LLM_MAX_RETRIES']
    )
    return QAPipeline(llm, vector_search, course_name=session_config['COURSE_NAME'])

# Enable Langsmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# FastAPI app initialization
app = FastAPI()

# Initialize the QA pipeline
qa_pipeline = setup_pipeline()

class QueryRequest(BaseModel):
    query: str

class Citation(BaseModel):
    url: str
    title: str
    
class Response(BaseModel):
    answer: str
    citations: Dict[int, Citation] 


@app.post("/ask", response_model=Response, summary="Anwer questions related to the course content.", 
        description="Submit a question to the QA pipeline and retrieve an answer with citations of relevant course content.")
async def ask_question(request: QueryRequest):
    query = request.query
    
    try:
        # Running the query through the QA pipeline
        response = qa_pipeline.run(query=query)
        answer = response["content"]
        citations = response["citation"]

        return Response(answer=answer, citations=citations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the query: {str(e)}")

