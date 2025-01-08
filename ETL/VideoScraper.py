
import argparse
import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from youtube_transcript_api import YouTubeTranscriptApi

# Set up logging
log_filename = "video_scraper.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode='a'),  # Append logs to the file
        logging.StreamHandler()  # Print logs to console
    ]
)
logger = logging.getLogger(__name__)

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
            
    def extract_submodule_iframe(self, submodule: Dict[str, Any]) -> Dict[str, Any]:
        """Extract iframes from a submodule."""
        try:
            soup = BeautifulSoup(submodule['html_content'], 'html.parser')
            extracted_iframes = []
            iframes = soup.find_all("iframe")
            for iframe in iframes:
                extracted_iframes.append({
                    "url": iframe.get("src", ""), 
                    "title": iframe.get("title", "")
                })
            
            logger.info(f"Extracted {len(extracted_iframes)} iframes from submodule: {submodule['title']}")
            return {
                "submodule_url": submodule['url'],
                "iframes": extracted_iframes,
            }
        except Exception as e:
            logger.error(f"Error extracting iframes from submodule {submodule['title']}: {e}")
            return {
                "submodule_url": submodule['url'],
                "iframes": [],
            }
        
    def process_module(self, module_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract iframes from all submodules in a module."""
        self._validate_module_data(module_data)
        module_title = module_data['module_title']
        module_iframes = []
        
        for submodule in module_data['submodule_data']:
            submodule_iframes = self.extract_submodule_iframe(submodule)
            module_iframes.append(submodule_iframes)
        
        logger.info(f"Extracted iframes from submodules in module: {module_title}")
                
        return module_iframes

class EchoTranscriptScraper:
    """Scrapes Echo360 video transcripts."""
    async def scrape_transcript(self, url: str, output_dir: str) -> Dict[str, str]:
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

                # Close the browser
                await browser.close()
                
                logger.info(f"Transcript of {url} saved to: {file_path}")
                
                transcript_metadata = {
                    "title": title,
                    "url": url,
                    "file_path": file_path,
                }

            except Exception as e:
                logger.error(f"Error scraping transcript from URL: {url}: {e}")
                self.failed_urls.append(url)
                transcript_metadata = {
                    "title": None,
                    "url": url,
                    "file_path": None,
                }
            
            return transcript_metadata

    async def scrape_transcripts(self, urls: List[str], output_dir: str, concurrency_limit: int = 5) -> List[Dict[str, str]]:
        """Scrapes the transcript of multiple URLs concurrently."""
        tasks = [self.scrape_transcript(url, output_dir) for url in urls]
        transcript_metadatas = await gather_with_concurrency(concurrency_limit, *tasks)
        return transcript_metadatas
    
    async def process_module_iframes(self, module_title: str, module_iframes: List[Dict[str, Any]], output_dir: str, concurrency_limit: int = 5) -> List[Dict[str, Any]]:
        """Processes the transcript of all Echo360 videos from a certain module."""
        module_transcript_metadatas  = []
        for submodule in module_iframes:
            iframes = submodule['iframes']
            echo_urls = [iframe['url'] for iframe in iframes if "echo360" in iframe['url']]
            if echo_urls:
                submodule_transcript_metadatas = await self.scrape_transcripts(echo_urls, output_dir, concurrency_limit)
                module_transcript_metadatas.append({   
                    "module_title": module_title,
                    "subsection": submodule['subsection'],
                    "submodule_title": submodule['title'],
                    "submodule_url": submodule['submodule_url'],
                    "transcript_metadatas": submodule_transcript_metadatas
                })
                
        # save module_transcript_metadatas to a file
        with open(f"{output_dir}/module_transcript_metadatas.json", 'w') as f:
            json.dump(module_transcript_metadatas, f, indent=4)
        logger.info(f"Saved Echo360 transcript metadata of {module_title} to: {output_dir}")

        return module_transcript_metadatas
    
class YoutubeTranscriptExtractor:
    """Extract YouTube video transcripts."""
    @staticmethod
    def _extract_youtube_embed_url(url: str) -> str:
        """Extract the youtube URL from an embed URL."""
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        video_url = query_params.get('url', None)[0]
        return video_url
    
    def extract_transcript(self, youtube_embed_url: str, title: str, output_dir: str) -> Dict[str, str]:
        """Get the transcripts of multiple YouTube videos."""
        try:
            youtube_url = self._extract_youtube_embed_url(youtube_embed_url)
            video_id = youtube_url.split("v=")[1]
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])
            
            # Save the transcript to a file
            file_path = f"{output_dir}/{sanitize_filename(title)}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(transcript, f, indent=4)
                
            logger.info(f"Transcript of {youtube_embed_url} saved to: {file_path}")
            
            return {
                "title": title,
                "url": youtube_url,
                "file_path": file_path,
            }
            
        except Exception as e:
            logger.error(f"Error extracting transcript from YouTube URL: {youtube_embed_url}: {e}")
            return {
                "title": title,
                "url": youtube_url,
                "file_path": None,
            }
    
    def process_module_iframes(self, module_title: str, module_iframes: List[Dict[str, Any]], output_dir: str) -> List[Dict[str, Any]]:
        """Processes the transcript of all YouTube videos from a certain module."""
        module_transcript_metadatas = []
        for submodule in module_iframes:
            iframes = submodule['iframes']
            submodule_transcript_metadatas = []
            for iframe in iframes:
                if "youtube" in iframe['url']:
                    transcript_metadata = self.extract_transcript(iframe['url'], iframe['title'], output_dir)
                    submodule_transcript_metadatas.append(transcript_metadata)
                    
            if submodule_transcript_metadatas:
                module_transcript_metadatas.append({
                    "module_title": module_title,
                    "subsection": submodule['subsection'],
                    "submodule_title": submodule['title'],
                    "submodule_url": submodule['submodule_url'],
                    "transcript_metadatas": submodule_transcript_metadatas
                })
            
        # save module_transcript_metadatas to a file
        with open(f"{output_dir}/module_transcript_metadatas.json", 'w') as f:
            json.dump(module_transcript_metadatas, f, indent=4)
        
        logger.info(f"Saved YouTube transcript of {module_title} metadata to: {output_dir}")
        
        return module_transcript_metadatas  

    
class VideoProcessor:
    """Processes video transcripts and extracts video iframes."""
    
    def __init__(self) -> None:
        self.iframe_extractor = IframeExtractor()
        self.echo_transcript_scraper = EchoTranscriptScraper()
        self.youtube_transcript_extractor = YoutubeTranscriptExtractor()
        
    async def process_module(self, module_data: Dict[str, Any], youtube_output_dir: str = '', echo360_output_dir: str = '') -> Dict[str, Any]:
        """This method processes video transcripts for a given module.
        If no output directory is provided for a platform, that platform's transcripts
        will not be processed.
        """
        try:
            if youtube_output_dir:
                if not os.path.exists(youtube_output_dir):
                    os.makedirs(youtube_output_dir)
                    logger.info(f"Created directory: {youtube_output_dir}")
            
            if echo360_output_dir:
                if not os.path.exists(echo360_output_dir):
                    os.makedirs(echo360_output_dir)
                    logger.info(f"Created directory: {echo360_output_dir}")
            
            module_title = module_data['module_title']        
            module_iframes = self.iframe_extractor.process_module(module_data)
            
            if youtube_output_dir:
                self.youtube_transcript_extractor.process_module_iframes(module_title, module_iframes, output_dir=youtube_output_dir)
            
            if echo360_output_dir:
                await self.echo_transcript_scraper.process_module_iframes(module_title, module_iframes, output_dir=echo360_output_dir)
            
            logger.info(f"Successfully processed module: {module_title}")
        except Exception as e:
            logger.error(f"Error processing module: {module_title}: {e}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Scrape video transcripts from YouTube and Echo360.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a single JSON file or a directory containing JSON files."
    )
    parser.add_argument(
        "--youtube-output-dir",
        help="Directory to save YouTube transcripts. If not provided, YouTube transcripts will not be processed."
    )
    parser.add_argument(
        "--echo360-output-dir",
        help="Directory to save Echo360 transcripts. If not provided, Echo360 transcripts will not be processed."
    )
    parser.add_argument(
        "--concurrency-limit",
        type=int,
        default=5,
        help="Maximum number of concurrent tasks for Echo360 scraping. Default is 5."
    )
    args = parser.parse_args()

    # Validate input path
    if not os.path.exists(args.input):
        parser.error(f"Input path does not exist: {args.input}")

    # Determine if the input is a file or a directory
    if os.path.isfile(args.input):
        json_files = [args.input]
        
    elif os.path.isdir(args.input):
        json_files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith('.json')]
        if not json_files:
            logger.error(f"No JSON files found in directory: {args.input}")
            return
    else:
        parser.error(f"Input path is neither a file nor a directory: {args.input}")

    # Process each JSON file
    video_processor = VideoProcessor()
    for json_file in json_files:
        logger.info(f"Processing file: {json_file}")

        with open(json_file, 'r', encoding='utf-8') as f:
            module_data = json.load(f)

        # Extract the module name from the filename
        module_name = os.path.splitext(os.path.basename(json_file))[0]

        # Set output directories
        youtube_output_dir = os.path.join(args.youtube_output_dir, module_name) if args.youtube_output_dir else None
        echo360_output_dir = os.path.join(args.echo360_output_dir, module_name) if args.echo360_output_dir else None

        # Process the module
        asyncio.run(video_processor.process_module(
            module_data,
            youtube_output_dir=youtube_output_dir,
            echo360_output_dir=echo360_output_dir
        ))

if __name__ == "__main__":
    main()
    # Example command from root project directory:
    # python ETL/VideoScraper.py --input "artifact/emgt605/module_content" --youtube-output-dir "artifact/emgt605/youtube" --echo360-output-dir "artifact/emgt605/echo360" --concurrency-limit 5
