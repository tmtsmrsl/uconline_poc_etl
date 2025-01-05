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
python ETL/CourseScraper.py --input_json artifact/emgt605/module_urls.json --ouput_dir artifact/emgt605/module_content
```
Note that if you are using Windows environment, you won't be able to run the code in Jupyter Notebook because of the Playwright incompatibility with the Windows event loop policy.

3. Process the HTML content to document dictionaries using `ETL/ContentProcessor.py`. The HTML content will be converted into a Markdown format, then split into several documents based on a predefined number of tokens. The document dictionaries also store the metadata of the documents. Please check the Section 1 of `ETL/vector_db_loading.ipynb` for an example.

4. Store the embeddings of the document dictionaries in a vector database. Please check the Section 2 of `ETL/vector_db_loading.ipynb` for an example. Note that I am using BGE-M3 for the dense embeddings, BM25 for the sparse embeddings, and Zilliz (cloud-managed Milvus) for the vector database. The trained sparse embeddings should be stored for future use.

## RAG 
Run the command below from the project root directory to start the Chainlit app on your local machine.
```bash
python -m chainlit run RAG/chainlit/app.py -w
```

## Chrome Extension
Rise Autoscroller is a chrome extension that can be used to automatically scroll through the relevant parts of the submodule content which is cited in the chatbot response. The extension is available on `autoscroll_extension` directory. To install the extension, follow the steps below:
1. Open the Extension Management page by navigating to `chrome://extensions`.
2. Enable Developer Mode by clicking the toggle switch on the top right corner of the page.
3. Click the "Load unpacked" button and select the `autoscroll_extension` directory.