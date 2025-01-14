import os
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException
from langchain_cerebras import ChatCerebras
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from RAG.utils.QAPipeline import QAPipeline
from RAG.utils.setup import initialize_vector_search, load_config, load_env_vars
from RAG.utils.ZillizVectorSearch import ZillizVectorSearch


def setup_pipeline(model_type: str, response_type: str, vector_search: ZillizVectorSearch) -> QAPipeline:
    """Setup the QA pipeline with pre-configured settings."""
    session_env = load_env_vars()
    session_config = load_config()

    if model_type == "llama-3.3":
        llm = ChatCerebras(
            api_key=session_env['CEREBRAS_API_KEY'],
            model="llama-3.3-70b",
            temperature=session_config['LLM_TEMPERATURE'],
            max_retries=session_config['LLM_MAX_RETRIES']
        )
    elif model_type == "gpt-4o":
        llm = ChatOpenAI(
            api_key=session_env['OPENAI_API_KEY'],
            model="gpt-4o",
            temperature=session_config['LLM_TEMPERATURE'],
            max_retries=session_config['LLM_MAX_RETRIES']
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid model type. Choose either 'llama-3.3' or 'gpt-4o'.")
    return QAPipeline(llm, vector_search, course_name=session_config['COURSE_NAME'], response_type=response_type)

# Enable Langsmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the vector search and store it in the app state."""
    session_env = load_env_vars()
    session_config = load_config()
    app.state.vector_search = initialize_vector_search(session_env, session_config)
    yield

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str
    model_type: str = "llama-3.3"
    response_type: str = "answer"

class Citation(BaseModel):
    url: str
    title: str
    content_type: str
    
class Response(BaseModel):
    answer: str
    citations: Dict[int, Citation] 


@app.post("/ask", response_model=Response, summary="Ask questions related to the course content.", 
        description="Submit a question to the QA pipeline and retrieve an answer or recommendation with citations of relevant course content.")
async def ask_question(request: QueryRequest):
    query = request.query
    model_type = request.model_type
    response_type = request.response_type
    
    try:
        # initializing the QA pipeline with the specified model type
        qa_pipeline = setup_pipeline(model_type=model_type, response_type=response_type, vector_search=app.state.vector_search)
        
        # Running the query through the QA pipeline
        response = qa_pipeline.run(query=query)
        answer = response["content"]
        citations = response["citation"]

        return Response(answer=answer, citations=citations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the query: {str(e)}")

