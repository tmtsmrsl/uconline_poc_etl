# PDF Extraction
PDF extraction is done using Docling, a MIT-licensed open-source package for document conversion. In the current implementation, the PDF file is converted to Markdown and images are not extracted. Use GPU-based environments for faster processing. One drawback of Docling is that all the headers are converted to a single level (<h2>). 