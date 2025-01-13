import json
import os
import re
from copy import deepcopy
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import webvtt
from deepmultilingualpunctuation import PunctuationModel
from langchain_core.documents import Document

from ETL.RecursiveTextSplitter import RecursiveTextSplitter


class TranscriptCleaner:
    def __init__(self, punctuation_model_name: str = "kredor/punctuate-all"):
        """
        Initializes the TranscriptCleaner class with an optional model for punctuation restoration.
        """
        self.model = PunctuationModel(model=punctuation_model_name)
    
    def clean_youtube_transcript(self, json_file: str) -> Dict:
        """
        Cleans a YouTube transcript by restoring punctuation and extracting metadata.
        """
        with open(json_file, "r") as f:
            transcript_fragments = json.load(f)

        # Make a temporary indexes based on the word count of unpunctuated text
        temp_indexes = []
        temp_text_list = []
        word_idx = 0

        for fragment in transcript_fragments:
            text = fragment['text']
            if text != "[Music]":
                temp_text_list.append(text)
                temp_indexes.append({"word_idx_start": word_idx, "start_time": fragment['start']})
                word_idx += len(text.split())

        temp_combined_texts = " ".join(temp_text_list)

        # Segment the text into sentences by adding punctuation 
        combined_texts = self.model.restore_punctuation(temp_combined_texts)

        text_list = combined_texts.split()
        indexes = []
        
        # Use the temporary indexes to find the character indexes for the punctuated text
        for idx in temp_indexes:
            text_up_to_idx = text_list[:idx['word_idx_start']]
            char_start = len(" ".join(text_up_to_idx))
            indexes.append({"char_start": char_start, "start_time": idx['start_time']})

        return combined_texts, indexes
    
    def _convert_vtt_time_to_seconds(self, vtt_timestamp) -> float:
        """
        Converts VTT timestamp to seconds.
        """
        hours, minutes, seconds, milliseconds = vtt_timestamp.to_tuple()
        duration = timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)
        return duration.total_seconds()

    def clean_echo360_transcript(self, vtt_file: str) -> Dict:
        """
        Cleans an Echo360 transcript by extracting text and metadata.
        """
        transcript_fragments = webvtt.read(vtt_file)

        indexes = []
        text_list = []
        char_idx = 0
        for fragment in transcript_fragments:
            # the echo360 transcript contains unnecessary newlines
            text = fragment.text.replace("\n", " ") + " "
            text_list.append(text)
            indexes.append({"char_start": char_idx, "start_time": self._convert_vtt_time_to_seconds(fragment.start_time)})
            char_idx += len(text)

        combined_texts = "".join(text_list)

        return combined_texts, indexes

class TranscriptDocProcessor:
    def __init__(self, text_splitter_options: Optional[dict] = None, return_dict: bool = False, punctuation_model_name: str = "kredor/punctuate-all", index_freq: int = 15) -> None:
        self.text_splitter_options = text_splitter_options
        self.return_dict = return_dict
        self.cleaner = TranscriptCleaner(punctuation_model_name)
        self.index_freq = index_freq
        
    @staticmethod
    def _filter_indexes(indexes: List[Dict], index_freq: int = 15) -> List[Dict]:
        """
        Filters the indexes to include only indexes that occur after every n seconds.
        """
        if not indexes:
            return []

        filtered_indexes = []
        filtered_indexes.append(indexes[0])
        last_start_time = indexes[0]['start_time']
        
        for index in indexes:
            if index['start_time'] >= last_start_time + index_freq:
                filtered_indexes.append(index)
                last_start_time = index['start_time']

        return filtered_indexes
    
    @staticmethod
    def _adjust_indexes(text: str, indexes: List[Dict]) -> List[Dict]:
        """
        Adjust the indexes to the start of the sentence that char_start is in.
        """
        new_indexes = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sent_starts = np.array([0])
        for sent in sentences:
            sent_starts = np.append(sent_starts, sent_starts[-1] + len(sent) + 1)
            
        for index in indexes:
            # find the sentence index that is before the char_start
            latest_sent_start = sent_starts[np.where(sent_starts <= index['char_start'])][-1]
            new_indexes.append({"start_time": index['start_time'], "char_start": latest_sent_start})
        return new_indexes
    
    def _clean_transcript(self, transcript_metadata: Dict, type: str, additional_metadata: Optional[Dict] = None) -> Dict:
        """
        Clean the transcript of a video and return a dictionary with the cleaned text and metadata.
        """
        temp_additional_metadata = deepcopy(additional_metadata) or {}
        if type == 'youtube':
            text, indexes = self.cleaner.clean_youtube_transcript(transcript_metadata['file_path'])
        elif type == 'echo360':
            text, indexes = self.cleaner.clean_echo360_transcript(transcript_metadata['file_path'])
        else:
            raise ValueError(f"Invalid transcript type: {type}")
        
        indexes = self._filter_indexes(indexes, self.index_freq)
        indexes = self._adjust_indexes(text, indexes)
        
        temp_additional_metadata.update({"index_metadata": indexes, 
                                    "video_title": transcript_metadata['title'], 
                                    "video_url": transcript_metadata['url'], 
                                    "video_desc": transcript_metadata['description']})
        transcript = {"content": text, "metadata": temp_additional_metadata}
        return transcript
        
    def _clean_submodule_transcripts(self, transcript_metadatas: List, type: str, additional_metadata: Optional[Dict] = None) -> List[Dict]:
        """Clean the transcripts of a submodule and add additional metadata."""
        temp_additional_metadata = deepcopy(additional_metadata) or {}
        
        cleaned_transcripts = [self._clean_transcript(transcript_metadata, type, temp_additional_metadata) for transcript_metadata in transcript_metadatas]
        return cleaned_transcripts

    def process_submodule_transcripts(self, submodule: Dict, additional_metadata: List[Dict] = None) -> List[Document]:
        """Process the transcripts of a submodule into a list of Documents."""
        temp_additional_metadata = deepcopy(additional_metadata) or {}
        cleaned_transcripts = []
        submodule_metadata = {
            'subsection': submodule['subsection'],
            'submodule_title': submodule['submodule_title'],
            'submodule_url': submodule['submodule_url']
        }
        temp_additional_metadata.update(submodule_metadata)
        cleaned_transcripts.extend(self._clean_submodule_transcripts(submodule['youtube_metadatas'], 'youtube', temp_additional_metadata))
        cleaned_transcripts.extend(self._clean_submodule_transcripts(submodule['echo360_metadatas'], 'echo360', temp_additional_metadata))
        
        text_splitter = RecursiveTextSplitter(**self.text_splitter_options)
        doc_splits = []
        
        for transcript in cleaned_transcripts:
            content = transcript['content']
            split_docs = text_splitter.split_text(content)
            split_docs = text_splitter.post_process_splits(split_docs, transcript['metadata'], self.return_dict)
            doc_splits.extend(split_docs)

        return doc_splits
    
    def process_module_transcripts(self, module_dir: str) -> List[Document]:
        """Process the transcripts of a module into a list of Documents."""
        cleaned_transcript_docs = []
        metadata_file = "metadata.json"

        with open(os.path.join(module_dir, metadata_file), "r") as f:
            module_metadata = json.load(f)

        for submodule in module_metadata['submodule_data']:
            additional_metadata = {'module_title': module_metadata['module_title']}
            cleaned_transcript_docs.extend(self.process_submodule_transcripts(submodule, additional_metadata))
            
        return cleaned_transcript_docs