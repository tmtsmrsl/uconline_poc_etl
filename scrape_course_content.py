import argparse
import asyncio
import json
import os
import re
from urllib.parse import urljoin

from playwright.async_api import async_playwright


async def gather_with_concurrency(limit, *tasks):
    """Asynchronously gather tasks with a concurrency limit."""
    if not tasks:
        raise ValueError("At least one task must be provided.")

    semaphore = asyncio.Semaphore(limit)

    async def sem_task(task):
        async with semaphore:
            return await task

    return await asyncio.gather(*(sem_task(task) for task in tasks))

async def scrape_submodule_urls(module_url):
    """Scrapes submodule URLs and titles from a given module URL."""
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        # Navigate to the page
        await page.goto(module_url)

        # Wait for the table of contents to load
        await page.wait_for_selector("nav.overview-list", timeout=10000)

        # Extract all the hyperlinks of the submodule
        hyperlinks = await page.locator("nav.overview-list a").all()

        # Initialize a list to store extracted data
        submodule_data = []

        # Loop through each hyperlink and extract the href and text
        for hyperlink in hyperlinks:
            href = await hyperlink.get_attribute('href')
            text = await hyperlink.inner_text()
            submodule_data.append({'url': urljoin(module_url, href), 
                                   'title': text.strip()})  
            
        # Close the browser
        await browser.close()

        return submodule_data
    
async def scrape_submodule_content(submodule_url):
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
        main_content_html = await main_content.inner_html()
            
        # Close the browser
        await browser.close()

        return main_content_html

async def process_submodule(submodule):
    """Processes a single submodule, by scraping its main content"""
    print(f"-- Scraping content of {submodule['title']} ({submodule['url']})")
    
    try:
        main_content_html = await scrape_submodule_content(submodule['url'])
    except Exception as e:
        print(f"-- Error scraping content for {submodule['title']} ({submodule['url']}): {e}")
        main_content_html = None
    return main_content_html

async def process_module(module, output_dir, con_limit=5):
    """Processes a single module, by scraping its submodules content"""
    module_name = module['module_name']
    module_url = module['module_url']
    print(f"Scraping submodules of {module_name} ({module_url})")
    
    try:
        submodule_data = await scrape_submodule_urls(module_url)
    except Exception as e:
        print(f"Error scraping submodule URLs for {module_name} ({module_url}): {e}")
        return # stop processing this module
    
    # Process submodules concurrently
    submodule_tasks = [
        process_submodule(submodule) for submodule in submodule_data
    ]
    processed_submodules = await gather_with_concurrency(con_limit, *submodule_tasks)

    # Collect results
    for i, result in enumerate(processed_submodules):
        submodule_data[i]['main_content_html'] = result
            
    # Save results
    save_output(
        {"module_url": module_url, "submodule_data": submodule_data},
        output_dir,
        module_name
    )

def load_input(input_json_path):
    """Load the input containing module URLs from a JSON file."""
    with open(input_json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def sanitize_filename(name):
    """Sanitize a string to be used as a filename."""
    filename = name.replace(" ", "_").replace(".", "_")
    filename = "".join(c for c in filename if c.isalnum() or c in ('_'))
    filename = re.sub(r"_+", "_", filename)
    filename = filename.lower()
    return filename

def save_output(data, output_dir, module_name):
    """Save the output data to a JSON file."""
    output_filename = sanitize_filename(module_name)
    output_path = os.path.join(output_dir, f"{output_filename}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {output_path}.")
    

async def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Scrape submodule URLs and their contents for each module.")

    parser.add_argument(
        "--input_json",
        required=True,
        help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save the output JSON files. The output files will be named as {module_name}.json."
    )
    args = parser.parse_args()

    # Load input data
    module_data = load_input(args.input_json)
    
    # Check if the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Process modules sequentially
    for module in module_data:
        await process_module(module, args.output_dir)

if __name__ == "__main__":
    asyncio.run(main())