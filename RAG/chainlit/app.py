import os
from typing import Dict, List, Tuple

import chainlit as cl
import requests
from chainlit.input_widget import Select

from RAG.utils.setup import load_config


async def send_initial_message():
    """Send an initial welcome message to the user."""
    session_config = cl.user_session.get("session_config")
    message_content = (
        f"Hi, I am UCOnline Copilot which can help you learn about {session_config['COURSE_NAME']}. " 
        "Ask me anything related to the course, and I will try to answer based on the course content!"
    )
    msg = cl.Message(content=message_content)
    await msg.send()

def _format_video_elem(source: Dict) -> cl.Text | cl.Video:
            """Generate the appropriate element based on the source of the video."""
            if "echo360" in source['url']:
                iframe_html = f"<html><iframe src={source['url']} width='100%' height='500px' frameborder='0'></iframe></html>"
                return cl.Text(name=source['title'], content=iframe_html, display="side")
            elif "youtube" in source['url']:
                return cl.Video(name=source['title'], url=source['url'], display="side")
            else:
                return ValueError(f"Unsupported video source: {source['url']}")
        
def _format_citations(citations: Dict[str, Dict]) -> Tuple[str, List]:
    """Formats citations and prepares video elements for rendering."""
    formatted_citations = ""
    video_elements = []
    if citations:
        for id, source in citations.items():
            if source['content_type'] == 'video_transcript':
                video_elements.append(_format_video_elem(source))
                formatted_citations += f"{id}. {source['title']}\n"
            else:
                formatted_citations += f"{id}. [{source['title']}]({source['url']})\n"  
    return formatted_citations, video_elements

async def send_answer_with_citations(answer: str, citations: Dict[str, Dict]):
    """
    Sends an answer along with formatted citations and associated video elements.
    """
    formatted_citations, video_elements = _format_citations(citations)
    answer_and_citations = f"{answer}\n\n**Sources:**\n{formatted_citations}"
    await cl.Message(content=answer_and_citations, elements=video_elements).send()
    
@cl.on_settings_update
async def update_settings(settings):
    """Update the selected model in the user session."""
    model_type = settings["model_type"]
    response_type = settings['response_type']
    cl.user_session.set("model_type", model_type)
    cl.user_session.set("response_type", response_type)

@cl.on_chat_start
async def start():
    cl.user_session.set("session_config", load_config())
    
    # Setup the settings interface and QAPipeline with the default model
    settings = await cl.ChatSettings(
        [
            Select(
                id="model_type",
                label="Model Type",
                values=["llama-3.3", "gpt-4o"],
                initial_index=0,
            ),
            Select(
                id="response_type",
                label="Response Type",
                values=["answer", "recommendation"],
                initial_index=0,
            ),
            
        ]
    ).send()
    await update_settings(settings)
    
    await send_initial_message()

@cl.on_message
async def main(message: cl.Message):
    # Prepare the request payload for the FastAPI endpoint
    payload = {
        "query": message.content,
        "model_type": cl.user_session.get("model_type"),
        "response_type": cl.user_session.get("response_type")
    }
    
    try:
        # Send the request to the FastAPI endpoint
        response = requests.post("http://127.0.0.1:8010/ask", json=payload)
        response.raise_for_status() 
        result = response.json()
        
        answer = result["answer"]
        citations = result["citations"]
        await send_answer_with_citations(answer, citations)
        
    except requests.exceptions.HTTPError as e:
        await cl.Message(content=f"An Error occured when fetching response from the FastAPI endpoint. Response: {e.response.text}").send()
    except Exception as e:
        await cl.Message(content=f"An unexpected error occurred: {str(e)}").send()
