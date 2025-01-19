# UCONLINE CHATBOT POC
A proof of concept for a chatbot that can be used to help students navigate through the course content for UC Online. The course used for this POC is EMGT605 - Sustainability Systems in Engineering and the module materials are available on the Articulate Rise platform. 

## Setup
It is recommended to use Python3.10 for this project to avoid any compatibility issues. All the required dependencies are listed in the `requirements.txt` file. Make a `.env` file in the project root directory and add the envinroment variables listed on the `.env.sample` file. 

## ETL
Instead of parsing the PDFs containing the screenshots of the module materials, I decided to scrape the content directly from the Articulate Rise platform using Playwright. The ETL process is as follows:
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

4. Process the HTML content and video transcripts to documents using `ETL/ContentProcessor.py` and `ETL/TranscriptProcessor.py` respectively. The documents will store the content in a format that is LLM-friendly, along with additional metadata. 

5. Store the embeddings of the documents in a vector database. 

If you want to do step 4 and 5 using Azure Search vector database and OpenAI text embeddings, you can use the `ETL/notebook/azure_vector_db_loading.ipynb` notebook. If you want to do step 4 and 5 using open-source tech stacks (Zilliz vector database and BGE-M3 embeddings), you can use the `ETL/notebook/zilliz_vector_db_loading.ipynb` notebook.

## RAG 
You can run the RAG pipeline either as a standalone FastAPI endpoint or bundled with the Chainlit app, which provides a chatbot interface for interaction. 

#### FastAPI endpoint
The FastAPI endpoint acts as a REST API for the RAG pipeline. Run the command below from the project root directory to start the FastAPI app on your local machine.
```bash
uvicorn RAG.fastapi.main:app --reload --port 8010
```

Example request to the FastAPI endpoint:
```bash
curl -X 'POST' \
  'localhost:8010/ask' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "How can we support indigenous sustainability?",
    "model_type": "llama-3.3",
    "response_type": "recommendation"
  }'
```
Currently, the available model types are `llama-3.3` and `gpt-4o`. Response types can be either `recommendation` (does not provide direct answer, but recommends relevant materials related to the question) or `answer` (provides direct answer to the question).

#### Chainlit app
If you want to interact with the FastAPI endpoint using a chatbot interface, you can run the command below from the project root directory to start the Chainlit app on your local machine.
```bash
python -m chainlit run RAG/chainlit/app.py -w --port 8000
```

The Chainlit app will run on `http://localhost:8000`. Note that the FastAPI endpoint should be running on `http://localhost:8010` for the Chainlit app to work properly.

Architecture of the RAG pipeline:  

![image](https://github.com/user-attachments/assets/b8d5e64e-f4e2-497e-9640-29f8b2584375)

1. Guardrail: Use LLM to filter out questions that violate the pre-defined guardrail criteria.
2. Retrieve: Retrieve the relevant documents from the vector database, combining results from both dense and sparse embeddings. These documents are then further reranked using the ColBERT model to ensure high-quality retrieval results. The retrieved documents are split based on the lesson blocks of the submodule content and formatted with an intermediate ID so the LLM can easily refer to it in step 3.
3. Generate: Use LLM to generate an answer with inline citation based on the documents retrieved from step 2 and the user’s question.
4. Format Answer: Refines the generated answer by reformatting the intermediate IDs and linking them to the original sources, ensuring traceability. Questions filtered out in step 1 still go through this step to ensure a consistent format for the final answer.

## Chrome Extension
Rise Autoscroller is a chrome extension that can be used to automatically scroll through the relevant parts of the submodule content which is cited in the chatbot response. The extension is available on `autoscroll_extension` directory. To install the extension, follow the steps below:
1. Open the Extension Management page by navigating to `chrome://extensions`.
2. Enable Developer Mode by clicking the toggle switch on the top right corner of the page.
3. Click the "Load unpacked" button and select the `autoscroll_extension` directory.
