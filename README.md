# UCONLINE CHATBOT POC
A proof of concept for a chatbot that can be used to help students navigate through the course content for UC Online. The course used for this POC is EMGT605 - Sustainability Systems in Engineering and the module materials are available on the Articulate Rise platform. 

## ETL
Instead of parsing the PDFs containing the screenshots of the module materials, I decided to scrape the content directly from the Articulate Rise platform using Playwright. The ETL process is as follows:
1. Save the URLs of the main module (not the submodule) in a JSON file. Below is an example of the JSON file structure.
```json
{"module_urls": [
    "https://rise.articulate.com/xy1", 
    "https://rise.articulate.com/qw2"]}
```

2. Scrape the module's submodule URLs and their respective HTML content using `CourseScraper.py`. Run the command below in your terminal.
```bash
playwright install
```
```bash
python ETL/CourseScraper.py --input_json course_materials/course_code/module_urls.json --ouput_dir course_materials/course_code/module_content
```
Note that if you are using Windows environment, you won't be able to run the code in Jupyter Notebook because of the Playwright incompatibility with the Windows event loop policy.

3. Process the HTML content to document dictionaries using `ContentProcessor.py`. The HTML content will be converted into a Markdown format, then split into several documents based on a predefined number of tokens. The document dictionaries also store the metadata of the documents. Please check the Section 1 of `vector_db_loading.ipynb` for an example.

4. Store the embeddings of the document dictionaries in a vector database. Please check the Section 2 of `vector_db_loading.ipynb` for an example. Note that I am using BGE-M3 for the embedding model and Zilliz (cloud-managed Milvus) for the vector database.
