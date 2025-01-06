import argparse
import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from playwright.async_api import async_playwright

# Set up logging
log_filename = "course_scraper.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode='a'),  # Append logs to the file
        logging.StreamHandler()  # Print logs to console
    ]
)
logger = logging.getLogger(__name__)

async def gather_with_concurrency(concurrency_limit: int = 5, *tasks: asyncio.Future) -> List[Any]:
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

class SubmoduleScraper:
    """Scrape and process submodule."""
    
    async def _scrape_submodule_content(self, submodule_url: str) -> str:
        """Scrapes the main content of a given submodule URL."""
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()

            # Navigate to the page
            await page.goto(submodule_url)

            # Wait for the main content to load
            await page.wait_for_selector("main.lesson-main", timeout=10000)

            # Extract the main content
            main_content = await page.query_selector("main.lesson-main")
            html_content = await main_content.inner_html()
                
            # Close the browser
            await browser.close()

            return html_content

    async def _process_submodule(self, submodule_url: str) -> Optional[str]:
        """Processes a single submodule by scraping its main content."""
        try:
            html_content = await self._scrape_submodule_content(submodule_url)
            logger.info(f"Successfully scraped content for {submodule_url}")
        except Exception as e:
            logger.error(f"Error scraping content for {submodule_url}: {e}")
            html_content = None
        return html_content
    
    async def process_submodules(self, submodule_urls: List[str], concurrency_limit: int = 5) -> List[Optional[str]]:
        """Processes a list of submodules concurrently."""
        tasks = [self._process_submodule(url) for url in submodule_urls]
        submodule_data = await gather_with_concurrency(concurrency_limit, *tasks)
        return submodule_data

class ModuleScraper:
    """Scrapes and process module along with its submodules."""

    def __init__(self) -> None:
        self.submodule_scraper = SubmoduleScraper()

    async def _scrape_module_structure(self, module_url: str) -> Dict[str, Any]:
        """Scrapes the module title, sections, and submodule URLs from a module URL."""
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            # Navigate to the page
            await page.goto(module_url)

            # Wait for the table of contents to load
            await page.wait_for_selector("nav.overview-list", timeout=10000)

            # Extract the module title
            module_title = await page.locator("header h1").inner_text()
            
            # Extract the module subsections
            module_subsections = await page.locator("nav.overview-list section").all()
            
            # Extract all the submodule hyperlinks within each module 
            submodule_data = []
            for subsection in module_subsections:
                subsection_title = await subsection.locator("h2").inner_text()
                hyperlinks = await subsection.locator("a").all()
                for hyperlink in hyperlinks:
                    href = await hyperlink.get_attribute('href')
                    text = await hyperlink.inner_text()
                    submodule_data.append(
                        {
                            'url': urljoin(module_url, href),
                            'title': text.strip(),
                            'subsection': subsection_title
                        }
                    )
            
            # Close the browser
            await browser.close()   
            
            return {
                "module_title": module_title,
                "module_url": module_url,
                "submodule_data": submodule_data
            }
            
    async def process_module(self, module_url: str, concurrency_limit: int) -> Optional[Dict[str, Any]]:
        """Processes a single module by scraping its metadata and submodules."""
        try:
            module_structure = await self._scrape_module_structure(module_url)
            logger.info(f"Successfully scraped submodule URLs for {module_url}")
        except Exception as e:
            logger.error(f"Error scraping submodule URLs for {module_url}: {e}")
            return  # Stop processing this module

        # Process submodules concurrently
        submodule_urls = [urljoin(module_url, submodule['url']) for submodule in module_structure['submodule_data']]
        processed_submodules = await self.submodule_scraper.process_submodules(submodule_urls, concurrency_limit)
        
        # Collect results
        for i, result in enumerate(processed_submodules):
            module_structure['submodule_data'][i]['html_content'] = result

        return module_structure
        
    @staticmethod
    def _load_input(input_json: str) -> List[str]:
        """Loads module URLs from a JSON file."""
        with open(input_json, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if 'module_urls' in data:
                    return data['module_urls']
                else:
                    raise ValueError("Missing 'module_urls' in the JSON.")
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON file.")

    @staticmethod
    def save_output(module_data: Dict[str, Any],
                    output_dir: str) -> None:
        """Saves the output data to a JSON file."""
        module_name = module_data['module_title']
        output_filename = sanitize_filename(module_name)
        output_path = os.path.join(output_dir, f"{output_filename}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(module_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}.")
    
    async def run(self, input_json: str, output_dir: Optional[str] = None, concurrency_limit: int = 5) -> List[Dict[str, Any]]:
        """Runs the scraping process for all modules."""
        module_urls = self._load_input(input_json)
        all_module_data = []
        
        # Create the output directory if specified and does not exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Process modules sequentially
        for module_url in module_urls:
            module_data = await self.process_module(module_url, concurrency_limit)
            if module_data:
                all_module_data.append(module_data)
                if output_dir:
                    self.save_output(module_data, output_dir)
        
        return all_module_data

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="For each module, scrape the module's submodule URLs and their respective content.")

    parser.add_argument(
        "--input_json",
        required=True,
        help="""Path to the input JSON file. It should have a key of 'module_urls' 
        and a list of URLS as the value. Each URL should be a string. 
        Example: {'module_urls': ['https://rise.articulate.com/xy1', 'https://rise.articulate.com/qw2']}."""
    )
    
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save the output JSON files. The output files will be named as {module_name}.json."
    )
    
    parser.add_argument(
        "--concurrency_limit",
        required=False,
        default=5,
        type=int,
        help="Concurrency limit for processing submodules. Default is 5.",
    )
    
    args = parser.parse_args()

    # Validate the input JSON file path
    if not os.path.isfile(args.input_json):
        parser.error(f"The input JSON file '{args.input_json}' does not exist.")
    
    # Validate concurrency limit
    if args.concurrency_limit <= 0:
        parser.error("The concurrency limit (--concurrency_limit) must be a positive integer.")
        
    # Run the scraper
    scraper = ModuleScraper()
    asyncio.run(scraper.run(input_json=args.input_json, output_dir=args.output_dir, concurrency_limit=args.concurrency_limit))

if __name__ == "__main__":
    main()