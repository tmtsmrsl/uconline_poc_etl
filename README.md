This repository implements the ETL pipeline for the EMGT605 chatbot, a proof of concept designed to help students navigate course content on UC Online. The pipeline scrapes module materials from the Articulate Rise platform and stores them in a vector database.

## Setup
It is recommended to use Python3.10 for this project to avoid any compatibility issues. All the required dependencies are listed in the `requirements.txt` file. Make a `.env` file in the project root directory and add the envinroment variables listed on the `.env.sample` file. 

## ETL
Instead of parsing the PDFs containing the screenshots of the module materials, I decided to scrape the content directly from the Articulate Rise platform using Playwright. This approach is more efficient as it eliminates the need for PDF export and relies on a rule-based method rather than computationally intensive ML models. It also produces better results since the rule-based method is more deterministic and ensures traceability of block sections.

The ETL process is as follows:
1. Save the URLs of the main module (not the submodule) in a JSON file. Below is an example of the JSON file structure.
```json
{"module_urls": [
    "https://rise.articulate.com/xy1", 
    "https://rise.articulate.com/qw2"]}
```

2. Scrape the module's submodule URLs and their respective HTML content using `ETL/CourseScraper.py`. Navigate to the project root directory and run the command below.
Install the Playwright dependencies.
```bash
playwright install
```
Then run the CourseScraper.py script with the input JSON file and the output directory as arguments.
```bash
python -m ETL.CourseScraper --input_json artifact/emgt605/module_urls.json --output_dir artifact/emgt605/html_content
```
This will save the HTML content of each module along with some metadata as an individual JSON file in the output directory. 

Note that if you are using Windows environment, you won't be able to run the code in Jupyter Notebook because of the Playwright incompatibility with the Windows event loop policy.

3. After scraping the HTML content, extract all the Youtube and Echo360 iframes and scrape their transcript using `ETL/TranscriptScraper.py`. Navigate to the project root directory and run the command below.
```bash
python -m ETL.TranscriptScraper --input artifact/emgt605/html_content --output-dir artifact/emgt605/transcripts
```
This will save the transcript and metadata (`metadata.json`) of videos for each module in the output directory. Youtube transcripts are saved as JSON files, and Echo360 transcripts are saved as VTT files in their own respective subdirectories.

4. Process the HTML content and video transcripts to documents using `ETL/ContentProcessor.py` and `ETL/TranscriptProcessor.py` respectively. The documents will store the content in a format that is LLM-friendly, along with additional metadata. The PunctuationModel is used to segment sentences from YouTube transcripts, as the original text lacks punctuation. 

5. Store the embeddings of the documents in a vector database. 

If you want to do step 4 and 5 using Azure Search vector database and OpenAI text embeddings, you can use the `ETL/notebook/azure_vector_db_loading.ipynb` notebook. If you want to do step 4 and 5 using open-source tech stacks (Zilliz vector database and BGE-M3 embeddings), you can use the `ETL/notebook/zilliz_vector_db_loading.ipynb` notebook.
