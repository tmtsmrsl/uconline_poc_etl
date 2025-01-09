from copy import deepcopy
from typing import Any, Dict, List

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RecursiveTextSplitter:
    def __init__(self, encoding_name: str = "o200k_base", chunk_token_size: int = 500, chunk_token_overlap: int = 50, separators: List[str] = ["\n\n", "\n", " ", ""]) -> None:
        self.encoding_name = encoding_name
        self.chunk_token_size = chunk_token_size
        self.chunk_token_overlap = chunk_token_overlap
        self.separators = separators
        
    def _add_overlap(self, doc_splits: List[Document]) -> List[Document]:
        """Adds overlapping tokens to preserve flow when splitting documents."""
        encoding = tiktoken.get_encoding(self.encoding_name)
        overlap_from_prev = ""
        for split in doc_splits:
            current_chunk = split.page_content
            split.page_content = overlap_from_prev + current_chunk
            split.metadata['start_index'] -= len(overlap_from_prev)
            current_chunk_tokens = encoding.encode(current_chunk)
            overlap_from_prev = encoding.decode(current_chunk_tokens[-self.chunk_token_overlap:])
            
        return doc_splits
    
    def split_text(self, text: str) -> List[Document]:
        """Split text content into smaller chunks based on token size."""
        
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=self.encoding_name,
            chunk_size=self.chunk_token_size,
            # we don't want to rely on the chunk overlap of the text splitter
            chunk_overlap=0,
            separators=self.separators,
            strip_whitespace=False,
            add_start_index=True
        )

        doc_splits = text_splitter.create_documents([text])
        doc_splits = self._add_overlap(doc_splits)
        return doc_splits
    
    @staticmethod
    def post_process_splits(doc_splits: List[Document], metadata: Dict[str, Any], return_dict: bool = False) -> List[Dict[str, Any]] | List[Document]:
        """Post-process the splits, adding metadata and optionally convert to dict format."""
        temp_splits = deepcopy(doc_splits)
        
        for split in temp_splits:
            split.metadata.update(metadata)
        
        if return_dict:
            temp_splits = [split.dict() for split in temp_splits]
            [split.pop('id', None) for split in temp_splits]
            
        return temp_splits