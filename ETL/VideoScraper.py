
import argparse
import asyncio
import json
import os
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from youtube_transcript_api import YouTubeTranscriptApi


async def gather_with_concurrency(concurrency_limit: int = 5, *tasks: asyncio.Future) -> list:
    """Asynchronously gather tasks with a concurrency limit."""
    if not tasks:
        raise ValueError("At least one task must be provided.")
    
    semaphore = asyncio.Semaphore(concurrency_limit)

    async def sem_task(task: asyncio.Future) -> Any:
        async with semaphore:
            return await task

    return await asyncio.gather(*(sem_task(task) for task in tasks))

def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be used as a filename."""
    filename = name.replace(" ", "_").replace(".", "_")
    filename = "".join(c for c in filename if c.isalnum() or c in ('_'))
    filename = re.sub(r"_+", "_", filename)
    filename = filename.lower()
    return filename

class IframeExtractor:
    """Extract the iframe from a module content JSON file."""
    def __init__(self) -> None:
        self.module_data = []
        self.failed_modules = []
        
    @staticmethod
    def _validate_module_data(module_data: Dict[str, Any]) -> None:
        """Validate module data."""
        if "module_title" not in module_data:
            raise ValueError("Module title not found in module data.")
        if "submodule_data" not in module_data:
            raise ValueError("Submodule data not found in module data.")
        for submodule in module_data["submodule_data"]:
            keys = ['url', 'subsection', 'title', 'html_content']
            if not all(key in submodule for key in keys):
                raise ValueError(f"Missing keys in submodule: {submodule}")
            
    def extract_submodule_iframe(self, submodule: Dict[str, Any], module_title: str) -> Dict[str, Any]:
        """Extract iframes from a submodule."""
        soup = BeautifulSoup(submodule['html_content'], 'html.parser')
        extracted_iframes = []
        iframes = soup.find_all("iframe")
        for iframe in iframes:
            extracted_iframes.append({"url": iframe.get("src", ""), 
                                    "title": iframe.get("title", "")})
            
        return {
            "submodule_url": submodule['url'],
            "iframes": extracted_iframes,
        }
        
    def process_module(self, module_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract iframes from all submodules in a module."""
        self._validate_module_data(module_data)
        module_title = module_data['module_title']
        module_iframes = []
        
        for submodule in module_data['submodule_data']:
            submodule_iframes = self.extract_submodule_iframe(submodule, module_title)
            module_iframes.append(submodule_iframes)
        
        return module_iframes

class EchoTranscriptScraper:
    """Scrapes Echo360 video transcripts."""

    def __init__(self) -> None:
        self.failed_urls = []

    async def _scrape_transcript(self, url: str, output_dir:str) -> Optional[str]:
        """Scrapes the transcript from a given URL."""
        async with async_playwright() as p:
            try:
                # Launch the browser
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                page = await context.new_page()

                # Navigate to the page
                await page.goto(url)

                # Wait for and click the download transcripts button
                await page.wait_for_selector("button[title='download transcripts']", timeout=5000)
                await page.click("button[title='download transcripts']")

                # Wait for and click the VTT format button
                await page.wait_for_selector("button[aria-label='VTT Format File']", timeout=5000)
                await page.click("button[aria-label='VTT Format File']", timeout=5000)
                
                # Wait for the download event
                download = await page.wait_for_event("download", timeout=5000)

                # Get the video title
                title = await page.title()
                
                # Save the file to the specified path
                file_path = f"{output_dir}/{sanitize_filename(title)}.vtt"
                await download.save_as(file_path)
                print(f"Transcript saved to: {file_path}")

                # Close the browser
                await browser.close()
                
                transcript_metadata = {
                    "title": title,
                    "url": url,
                    "file_path": file_path,
                }

            except Exception as e:
                print(f"Error scraping transcript from {url}: {e}")
                self.failed_urls.append(url)
                transcript_metadata = {
                    "title": None,
                    "url": url,
                    "file_path": None,
                }
            
            return transcript_metadata

    async def scrape_urls(self, urls: list[str], output_dir: str, concurrency_limit: int = 5) -> list[str]:
        """Scrapes the transcript of multiple URLs concurrently."""
        tasks = [self._scrape_transcript(url, output_dir) for url in urls]
        transcript_metadatas = await gather_with_concurrency(concurrency_limit, *tasks)
        return transcript_metadatas
    
    async def process_module_iframes(self, module_iframes: List[Dict[str, Any]], output_dir, concurrency_limit: int = 5) -> List[Dict[str, Any]]:
        """Processes the transcript of all Echo360 videos from a certain module."""
        module_transcript_metadatas  = []
        for submodule in module_iframes:
            iframes = submodule['iframes']
            echo_urls = [iframe['url'] for iframe in iframes if "echo360" in iframe['url']]
            if echo_urls:
                submodule_transcript_metadatas = await self.scrape_urls(echo_urls, output_dir, concurrency_limit)
                module_transcript_metadatas.append({   
                    "submodule_url": submodule['submodule_url'],
                    "transcript_metadatas": submodule_transcript_metadatas
                })
        # save module_transcript_metadatas to a file
        with open(f"{output_dir}/module_transcript_metadatas.json", 'w') as f:
            json.dump(module_transcript_metadatas, f, indent=4)
    
class YoutubeTranscriptExtractor:
    """Extract YouTube video transcripts."""
    def __init__(self) -> None:
        self.failed_urls = []
        
    @staticmethod
    def extract_youtube_embed_url(url: str) -> str:
        """Extract the youtube URL from an embed URL."""
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        video_url = query_params.get('url', None)[0]
        return video_url
    
    def get_youtube_transcript(self, youtube_embed_url, title, output_dir):
        """Get the transcripts of multiple YouTube videos."""
        youtube_url = self.extract_youtube_embed_url(youtube_embed_url)
        video_id = youtube_url.split("v=")[1]
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])
            # Save the transcript to a file
            file_path = f"{output_dir}/{sanitize_filename(title)}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(transcript, f, indent=4)
                print(f"Transcript saved to: {file_path}")
            return {
                "title": title,
                "url": youtube_url,
                "file_path": file_path,
            }
            
        except Exception as e:
            print(f"Error getting transcript for video {video_id}: {e}")
            return {
                "title": title,
                "url": youtube_url,
                "file_path": None,
            }
    
    def process_module_iframes(self, module_iframes: List[Dict[str, Any]], output_dir: str) -> List[Dict[str, Any]]:
        """Processes the transcript of all YouTube videos from a certain module."""
        module_transcript_metadatas = []
        for submodule in module_iframes:
            iframes = submodule['iframes']
            submodule_transcript_metadatas = []
            for iframe in iframes:
                if "youtube" in iframe['url']:
                    transcript_metadata = self.get_youtube_transcript(iframe['url'], iframe['title'], output_dir)
                    submodule_transcript_metadatas.append(transcript_metadata)
                    
            if submodule_transcript_metadatas:
                module_transcript_metadatas.append({
                    "submodule_url": submodule['submodule_url'],
                    "transcript_metadatas": submodule_transcript_metadatas
                })
            
        # save module_transcript_metadatas to a file
        with open(f"{output_dir}/module_transcript_metadatas.json", 'w') as f:
            json.dump(module_transcript_metadatas, f, indent=4)
    
class VideoProcessor:
    """Processes video transcripts and extracts video iframes."""
    
    def __init__(self) -> None:
        self.iframe_extractor = IframeExtractor()
        self.echo_transcript_scraper = EchoTranscriptScraper()
        self.youtube_transcript_extractor = YoutubeTranscriptExtractor()
        
    async def process_module(self, module_data: Dict[str, Any], youtube_output_dir: str = '', echo360_output_dir: str = '') -> Dict[str, Any]:
        """Processes the video transcripts and iframes of a module."""
        if youtube_output_dir:
            if not os.path.exists(youtube_output_dir):
                os.makedirs(youtube_output_dir)
        
        if echo360_output_dir:
            if not os.path.exists(echo360_output_dir):
                os.makedirs(echo360_output_dir)
                
        module_iframes = self.iframe_extractor.process_module(module_data)
        
        if youtube_output_dir:
            self.youtube_transcript_extractor.process_module_iframes(module_iframes, output_dir=youtube_output_dir)
        
        if echo360_output_dir:
            await self.echo_transcript_scraper.process_module_iframes(module_iframes, output_dir=echo360_output_dir)
        
if __name__ == "__main__":
    # Load the module data
    module_content_dir = "artifact/emgt605/module_content"
    module_content_jsons = os.listdir(module_content_dir)
    
    for module_content_json in module_content_jsons:
        with open(f"{module_content_dir}/{module_content_json}", 'r', encoding='utf-8') as f:
            module_data = json.load(f)
        
        # extract the json filename without the format
        module_name = os.path.basename(module_content_json).split(".")[0]
        youtube_output_dir = f"artifact/emgt605/video_transcript/youtube/{module_name}"
        echo360_output_dir = f"artifact/emgt605/video_transcript/echo360/{module_name}"
        
        # Process the module data
        video_processor = VideoProcessor()
        asyncio.run(video_processor.process_module(module_data, youtube_output_dir=youtube_output_dir, echo360_output_dir=echo360_output_dir))
        