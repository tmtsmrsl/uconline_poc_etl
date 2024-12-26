import json
import os
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional

import tiktoken
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from markdownify import MarkdownConverter


class CustomMDConverter(MarkdownConverter):
    """
    Inherit from MarkdownConverter to customize the conversion of HTML to Markdown.
    """
    def convert_iframe(self, el, text, convert_as_inline):
        """
        Markdownify does not support iframe extraction. This method will preserve the iframe as HTML tags in the markdown content.
        """
        title = el.attrs.get('title', '')
        src = el.attrs.get('src', '')
        return f'<iframe title="{title}" src="{src}"></iframe>\n\n'
    
    def _convert_hn(self, n, el, text, convert_as_inline):
        """
        Override the default implementation of heading tags conversion by markdownify. This implementation will preserve the heading style as HTML tags if the heading_style option is set to "html". 
        """
        if self.options['heading_style'].lower() == "html":
            text = text.strip()
            return f"\n<h{n}>{text}</h{n}>\n\n"
        else:
            return super()._convert_hn(n, el, text, convert_as_inline)

class ContentHTMLProcessor:
    """
    Class to preprocess HTML content to make it more suitable for markdown conversion.
    """
    def __init__(self, html: str) -> None:
        self.soup = BeautifulSoup(html, 'html.parser')
    
    def exclude_elements(self, css_selector: Optional[str] = None) -> 'ContentHTMLProcessor':
        """
        Excludes elements matching the given CSS selector(s) from the HTML.
        """
        if css_selector:
            excluded_elements = self.soup.select(css_selector)
            for element in excluded_elements:
                element.decompose()
        return self
    
    def modify_divs_spacing(self) -> 'ContentHTMLProcessor':
        """Insert <br> tags after each <div> element to ensure all divs are separated by a newline. The current implementation of markdownify does not handle this well.
        """
        for div in self.soup.select('div'):
            div.insert_after(self.soup.new_tag('br'))
        return self

    def extract_font_size(self) -> 'ContentHTMLProcessor':
        """Extract font-size from <span> elements and include it in the text. This will be used as a metadata to help preserve content structure during text splitting. Currently it is assumed that the font-size is always in rem units.  
        """
        spans_with_font_size = self.soup.select('span[style*="font-size"]')
        for span in spans_with_font_size:
            style = span.get('style', '')
            font_size_match = re.search(r'font-size:\s*([\d.]+)rem', style)
            
            if font_size_match:
                # Add the font-size as metadata before and after the span tag
                rem_value = float(font_size_match.group(1))
                metadata_str = f'[[rem: {rem_value:.2f}]]'
                span.insert_before(metadata_str)
                
        return self

    def get_html(self) -> str:
        """Return the modified HTML as a string."""
        return str(self.soup)

class ContentMDFormatter:
    """
    Class to format submodule content from HTML to Markdown.
    """
    def __init__(self, submodule_html: str, excluded_elements_css: Optional[str] = None, **md_converter_options: Any) -> None:
        self.submodule_html = submodule_html
        self.md_converter_options = {"heading_style": "html"}
        self.md_converter_options.update(md_converter_options) 
        self.excluded_elements_css = excluded_elements_css

    @staticmethod
    def clean_spacing(text: str) -> str:
        """Strip excessive newlines and unnecessary spaces."""
        text = re.sub(r'\s*\n\s*', '\n', text)
        text = re.sub(r'\n+', '\n\n', text)
        text = re.sub(r'(\n\n>)+', '\n\n>', text)
        return text

    def _html_to_md(self, html: str) -> str:
        """Preprocess and convert HTML content to Markdown."""
        # Preprocess the HTML content
        html_processor = ContentHTMLProcessor(html)
        html_processor.modify_divs_spacing().exclude_elements(self.excluded_elements_css)
        processed_html = html_processor.get_html()
        
        # Convert the processed HTML content to Markdown
        md_converter = CustomMDConverter(**self.md_converter_options)
        md_content = md_converter.convert(processed_html)
        md_content = self.clean_spacing(md_content)
        return md_content
    
    def _process_lesson_blocks(self, lesson_block_divs: List[Any]) -> List[Dict[str, Any]]:
        """Process lesson blocks into Markdown with metadata."""
        data_blocks = []
        offset = 1
        
        for div in lesson_block_divs:
            data_block_id = div.get('data-block-id', '')
            md_content = self._html_to_md(str(div))
            
            # check if md_content contains characters other than whitespace
            if md_content.strip() != "":
                char_length = len(md_content)
                data_blocks.append({
                    "data_block_id": data_block_id,
                    "md_content": md_content,
                    "char_start": offset,
                    "char_end": offset + char_length - 1
                })
                offset += char_length
        
        return data_blocks

    def to_md(self, split_blocks: bool = True) -> List[Dict[str, Any]] | str:
        """Convert HTML content to Markdown. If split_blocks is True, the content will be split into blocks based on the lesson block divs."""
        if split_blocks:
            soup = BeautifulSoup(self.submodule_html, 'html.parser')
            lesson_block_divs = soup.select('section.blocks-lesson > div')
            return self._process_lesson_blocks(lesson_block_divs)
        else:
            return self._html_to_md(self.submodule_html)


class ContentDocProcessor:
    """
    Class to process course content from HTML to LangChain document, which will be used for embedding.
    """

    def __init__(self, excluded_elements_css: str, chunk_token_size: int = 500, chunk_token_overlap: int = 50, encoding_name: str = "o200k_base") -> None:
        self.excluded_elements_css = excluded_elements_css
        self.encoding_name = encoding_name
        self.chunk_token_size = chunk_token_size
        self.chunk_token_overlap = chunk_token_overlap

    def _combine_blocks(self, lesson_blocks: List[Dict[str, Any]]) -> str:
        """Combine lesson blocks into a single Markdown string."""
        return "".join(block['md_content'] for block in lesson_blocks)
    
    def _add_overlap(self, splits: List[Any], chunk_token_overlap: int) -> List[Any]:
        """Adds overlapping tokens to preserve flow when splitting documents."""
        encoding = tiktoken.get_encoding(self.encoding_name)
        overlap_from_prev = ""
        for split in splits:
            current_chunk = split.page_content
            split.page_content = overlap_from_prev + current_chunk
            current_chunk_tokens = encoding.encode(current_chunk)
            overlap_from_prev = encoding.decode(current_chunk_tokens[-chunk_token_overlap:])
        return splits
    
    def _split_text(self, combined_md: str) -> List[Any]:
        """Split combined Markdown content into smaller chunks based on token size."""
        h_tags = ['<h1>', '<h2>', '<h3>']
        whitespace_separators = ["\n\n\n\n", "\n\n", "\n", " ", ""]
        separators = h_tags + whitespace_separators
        
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=self.encoding_name,
            chunk_size=self.chunk_token_size,
            # we don't want to rely on the chunk overlap of the text splitter
            chunk_overlap=0,
            separators=separators,
            strip_whitespace=False,
            add_start_index=True
        )

        splits = text_splitter.create_documents([combined_md])
        splits = self._add_overlap(splits, self.chunk_token_overlap)
        for split in splits:
            split.page_content = ContentMDFormatter.clean_spacing(split.page_content)
        return splits


    def _post_process_splits(self, splits: List[Any], metadata: Dict[str, Any]) -> List[Any]:
        """Post-process the splits, adding metadata and formatting content."""
        for split in splits:
            split.metadata.update(metadata)
            split.metadata['split_char_start'] = split.metadata['start_index'] + 1
            split.metadata['split_char_end'] = split.metadata['split_char_start'] + len(split.page_content) - 1
            del split.metadata['start_index']
            
            # Adding this prefix will help preserve the contextual information of the split
            prefix = f"This is part of the lesson: {metadata['module_title']} - {metadata['subsection']}: {metadata['submodule_title']}"
            split.page_content = f"{prefix}\n\n{split.page_content.strip()}"
        return splits

    def _process_submodule(self, submodule: Dict[str, Any], module_title: str) -> Dict[str, Any]:
        """Process lesson blocks for a submodule into LangChain document splits."""
        temp_submodule = deepcopy(submodule)
        
        # Convert HTML content to Markdown blocks
        md_lesson_blocks = ContentMDFormatter(temp_submodule['html_content'], self.excluded_elements_css).to_md(split_blocks=True)
        
        # Generate block ranges for metadata
        data_block_ranges = {
            block['data_block_id']: {
                'char_start': block['char_start'],
                'char_end': block['char_end']
            }
            for block in md_lesson_blocks
        }
        
        # Combine the Markdown blocks
        combined_md = self._combine_blocks(md_lesson_blocks)
        
        # Update submodule with lesson blocks and document splits
        temp_submodule['md_lesson_blocks'] = md_lesson_blocks
        temp_submodule['doc_splits'] = self._split_text(combined_md)
        
        metadata = {
            "data_block_ranges": data_block_ranges,
            "module_title": module_title.title(),
            "subsection": temp_submodule['subsection'].title(),
            "submodule_title": temp_submodule['title'].title(),
            "submodule_url": temp_submodule['url']
        }
        
        # Post-process splits with metadata and content formatting
        temp_submodule['doc_splits'] = self._post_process_splits(temp_submodule['doc_splits'], metadata)
        return temp_submodule
    
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
    
    def process_module(self, module_data: Dict[str, Any]) -> List[Any]:
        """Process all submodules in a module into LangChain documents."""
        self._validate_module_data(module_data)
        module_title = module_data['module_title']
        docs = []
        
        for submodule in module_data['submodule_data']:
            submodule = self._process_submodule(submodule, module_title)
            docs.extend(submodule['doc_splits'])
        return docs

    def run(self, input_json: str, output_dir: Optional[str] = None) -> List[Any]:
        """Process all submodules in the input JSON file into LangChain documents and save the output."""
        with open(input_json, "r", encoding="utf-8") as f:
            module_data = json.load(f)

        # Create the output directory if specified and does not exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Process the submodules
        docs = self.process_module(module_data)

        if output_dir:
            self.save_output(output_dir)
            
        return docs