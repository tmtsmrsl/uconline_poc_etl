import os
from typing import Dict

import joblib
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from RAG.utils.ColBERTReranker import ColBERTReranker
from RAG.utils.QAPipeline import QAPipeline
from RAG.utils.ZillizVectorSearch import ZillizVectorSearch

# Load environment variables
load_dotenv()

# FastAPI app initialization
app = FastAPI()

# Load environment variables from .env
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_USER = os.getenv("ZILLIZ_USER")
ZILLIZ_PASSWORD = os.getenv("ZILLIZ_PASSWORD")
COLLECTION_NAME = "emgt_605_bge_bm25_500_50"
SPARSE_EMBEDDINGS_PATH = "artifact/emgt605/sparse_embeddings.joblib"
COLBERT_MODEL_NAME = "answerdotai/answerai-colbert-small-v1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COURSE_NAME = "Sustainability Systems in Engineering"

# Setup langsmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialize the models and the QA pipeline
dense_embeddings = BGEM3EmbeddingFunction(use_fp16=False, device=DEVICE, return_dense=True, return_sparse=False)
sparse_embeddings = joblib.load(SPARSE_EMBEDDINGS_PATH)
colbert_reranker = ColBERTReranker(model_name=COLBERT_MODEL_NAME)
vector_search = ZillizVectorSearch(ZILLIZ_USER, ZILLIZ_PASSWORD, ZILLIZ_URI, COLLECTION_NAME, sparse_embeddings, dense_embeddings, colbert_reranker)
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o", temperature=0, max_retries=3)

qa_pipeline = QAPipeline(llm, vector_search, course_name=COURSE_NAME)

class QueryRequest(BaseModel):
    query: str

class Citation(BaseModel):
    url: str
    title: str
    
class Response(BaseModel):
    answer: str
    citations: Dict[int, Citation] 

@app.post("/ask", response_model=Response)
async def ask_question(request: QueryRequest):
    query = request.query
    
    # Running the query through the QA pipeline
    response = qa_pipeline.run(query=query)
    answer = response["content"]
    citations = response["citation"]
    
    return Response(answer=answer, citations=citations)

