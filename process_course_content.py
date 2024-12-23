from bs4 import BeautifulSoup
from markdownify import MarkdownConverter
import re

class ContentMDConverter(MarkdownConverter):
    """
    Custom MarkdownConverter class to convert HTML to markdown content.
    """
    def convert_iframe(self, el, text, convert_as_inline):
        """
        Markdownify does not support iframe extraction. This method will preserve the iframe as HTML tags in the markdown content.
        """
        title = el.attrs.get('title', '')
        src = el.attrs.get('src', '')
        return f'<iframe title="{title}" src="{src}">\n\n'
    
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
    def __init__(self, html):
        self.soup = BeautifulSoup(html, 'html.parser')
        
    def modify_divs_spacing(self):
        """Insert <br> tags after each <div> element to ensure all divs are separated by a newline. The current implementation of markdownify does not handle this well.
        """
        for div in self.soup.find_all('div'):
            div.insert_after(self.soup.new_tag('br'))
        return self

    def extract_font_size(self):
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

    def get_html(self):
        """Return the modified HTML as a string."""
        return str(self.soup)

class ContentFormatter:
    """
    Class to format course content from HTML to Markdown.
    """
    def __init__(self, html, **md_converter_options):
        self.html = html
        self.md_converter_options = {"heading_style": "html"}
        self.md_converter_options.update(md_converter_options) 

    @staticmethod
    def _clean_spacing(text):
        """Strip excessive newlines and unnecessary spaces."""
        text = re.sub(r'\s*\n\s*', '\n', text)
        text = re.sub(r'\n+', '\n\n', text)
        text = re.sub(r'(\n\n>)+', '\n\n>', text)
        return text.lstrip()    
    
    def _html_to_md(self, html):
        """Preprocess and convert HTML content to Markdown."""
        # Preprocess the HTML content
        html_processor = ContentHTMLProcessor(html)
        html_processor.modify_divs_spacing().extract_font_size()
        processed_html = html_processor.get_html()
        
        # Convert the processed HTML content to Markdown
        md_converter = ContentMDConverter(**self.md_converter_options)
        md_content = md_converter.convert(processed_html)
        md_content = self._clean_spacing(md_content)
        return md_content
    
    def _process_lesson_blocks(self, lesson_block_divs):
        """Process lesson blocks into Markdown with metadata."""
        data_blocks = []
        offset = 1
        
        for div in lesson_block_divs:
            data_block_id = div.get('data-block-id', '')
            md_content = self._html_to_md(str(div))
            char_length = len(md_content)
            
            data_blocks.append({
                "data_block_id": data_block_id,
                "md_content": md_content,
                "char_start": offset,
                "char_end": offset + char_length - 1
            })
            offset += char_length
        
        return data_blocks
    
    def to_md(self, split_blocks=True):
        """Convert HTML content to Markdown. If split_blocks is True, the content will be split into blocks based on the lesson block divs."""
        if split_blocks:
            soup = BeautifulSoup(self.html, 'html.parser')
            lesson_block_divs = soup.find('section', class_='blocks-lesson').find_all('div', recursive=False)
            return self._process_lesson_blocks(lesson_block_divs)
        else:
            return self._html_to_md(self.html)

