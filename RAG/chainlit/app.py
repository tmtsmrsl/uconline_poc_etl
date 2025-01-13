import os
from typing import Dict, List

import chainlit as cl
import requests
from chainlit.input_widget import Select

from RAG.utils.setup import load_config


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
async def update_model(settings):
    """Update the selected model in the user session."""
    selected_model = settings["Model"]
    cl.user_session.set("model_type", selected_model)

@cl.on_chat_start
async def start():
    cl.user_session.set("session_config", load_config())
    
    # Setup the settings interface and QAPipeline with the default model
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Model",
                values=["llama-3.3", "gpt-4o"],
                initial_index=0,
            )
        ]
    ).send()
    await update_model(settings)
    
    await send_initial_message()

@cl.on_message
async def main(message: cl.Message):
    model_type = cl.user_session.get("model_type") 
    session_config = cl.user_session.get("session_config")

    # Prepare the request payload for the FastAPI endpoint
    payload = {
        "query": message.content,
        "model_type": model_type,
        "course_name": session_config["COURSE_NAME"]
    }
    
    try:
        # Send the request to the FastAPI endpoint
        response = requests.post("http://127.0.0.1:8010/ask", json=payload)
        response.raise_for_status() 
        result = response.json()
        
        answer = result["answer"]
        citations = result["citations"]
        citation_elements = format_citation_elements(citations)
        
        await cl.Message(content=answer, elements=citation_elements).send()
    except requests.exceptions.HTTPError as e:
        await cl.Message(content=f"An Error occured when fetching response from the FastAPI endpoint. Response: {e.response.text}").send()
    except Exception as e:
        await cl.Message(content=f"An unexpected error occurred: {str(e)}").send()
