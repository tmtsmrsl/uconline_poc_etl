import os
from typing import Dict, List

import chainlit as cl
from chainlit.input_widget import Select
from langchain_cerebras import ChatCerebras
from langchain_openai import ChatOpenAI

from RAG.utils.QAPipeline import QAPipeline
from RAG.utils.setup import initialize_vector_search, load_config, load_env_vars


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
    # Load environment variables and configuration settings
    cl.user_session.set("session_env", load_env_vars())
    cl.user_session.set("session_config", load_config())
    
    # intialize vector search
    vector_search = initialize_vector_search(cl.user_session.get("session_env"), cl.user_session.get("session_config"))    
    cl.user_session.set("vector_search", vector_search)
    
    # Enable the tracing feature of Langsmith
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    
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

