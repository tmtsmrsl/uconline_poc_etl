import os
from typing import Dict

from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from RAG.utils.QAPipeline import QAPipeline
from RAG.utils.setup import initialize_vector_search, load_config, load_env_vars


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

