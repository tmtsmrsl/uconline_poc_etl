import os

import chainlit as cl
import joblib
import torch
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from RAG.utils.ColBERTReranker import ColBERTReranker
from RAG.utils.QAPipeline import QAPipeline
from RAG.utils.ZillizVectorSearch import ZillizVectorSearch


@cl.on_chat_start
async def on_chat_start():
    load_dotenv()
    
    # Utilize GPU to load and infer the embedding model if available 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # This is the parameter to connect to the Zilliz vector database
    ZILLIZ_URI = os.getenv("ZILLIZ_URI")
    ZILLIZ_USER = os.getenv("ZILLIZ_USER")
    ZILLIZ_PASSWORD = os.getenv("ZILLIZ_PASSWORD")
    COLLECTION_NAME = "emgt_605_bge_bm25_500_50"

    # The file path to load the sparse embeddings
    SPARSE_EMBEDDINGS_PATH = "artifact/emgt605/sparse_embeddings.joblib"

    # The name of ColBERT model that will be used as a reranker
    COLBERT_MODEL_NAME = "answerdotai/answerai-colbert-small-v1"

    # The API key to access the LLM
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # The name of the course that the AI assistant will help with
    COURSE_NAME = "Sustainability Systems in Engineering"
    
    # Enable the tracing feature of Langsmith
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

    dense_embeddings = BGEM3EmbeddingFunction(use_fp16=False, device=DEVICE, return_dense=True, return_sparse=False)
    sparse_embeddings = joblib.load(SPARSE_EMBEDDINGS_PATH)
    colbert_reranker = ColBERTReranker(model_name=COLBERT_MODEL_NAME)
    vector_search = ZillizVectorSearch(ZILLIZ_USER, ZILLIZ_PASSWORD, ZILLIZ_URI, COLLECTION_NAME, sparse_embeddings, dense_embeddings, colbert_reranker) 
    
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o", temperature=0, max_retries=3)
    
    qa_pipeline = QAPipeline(llm, vector_search, course_name=COURSE_NAME)
    msg = cl.Message(content=f"Hi, I am UCOnline AI Assitant which can help you learn {COURSE_NAME}.")
    await msg.send()
    
    cl.user_session.set("qa_pipeline", qa_pipeline)

@cl.on_message
async def main(message: cl.Message):
    qa_pipeline = cl.user_session.get("qa_pipeline") 

    response = qa_pipeline.run(query=message.content)
    answer = response["content"]
    citations = response["citation"]
    
    if citations:
        formatted_citations = ""
        for id, source in citations.items():
            formatted_citations += f"{id}. [{source['title']}]({source['url']})\n"
        elements = [
            cl.Text(name="Sources", content=formatted_citations, display="inline")
            ]
    else:
        elements = []

    await cl.Message(content=answer, elements=elements).send() 