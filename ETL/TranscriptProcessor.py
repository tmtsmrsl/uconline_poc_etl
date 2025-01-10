import json
import os
from datetime import timedelta
from typing import Callable, Dict, List, Optional

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

        # Make a temporary index based on the word count of unpunctuated text
        temp_index = []
        temp_text_list = []
        word_idx = 0

        for fragment in transcript_fragments:
            text = fragment['text']
            if text != "[Music]":
                temp_text_list.append(text)
                temp_index.append({"word_idx_start": word_idx, "start_time": fragment['start']})
                word_idx += len(text.split())

        temp_combined_texts = " ".join(temp_text_list)

        # Segment the text into sentences by adding punctuation 
        combined_texts = self.model.restore_punctuation(temp_combined_texts)

        text_list = combined_texts.split()
        index = []
        
        # Use the temporary index to find the character index for the punctuated text
        for idx in temp_index:
            text_up_to_idx = text_list[:idx['word_idx_start']]
            char_start = len(" ".join(text_up_to_idx))
            index.append({"char_start": char_start, "start_time": idx['start_time']})

        return combined_texts, index
    
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

        index = []
        text_list = []
        char_idx = 0
        for fragment in transcript_fragments:
            # the echo360 transcript contains unnecessary newlines
            text = fragment.text.replace("\n", " ")
            text_list.append(text)
            index.append({"char_start": char_idx, "start_time": self._convert_vtt_time_to_seconds(fragment.start_time)})
            char_idx += len(text)

        combined_texts = " ".join(text_list)

        return combined_texts, index

class TranscriptDocProcessor:
    def __init__(self, text_splitter_options: Optional[dict] = None, return_dict: bool = False, punctuation_model_name: str = "kredor/punctuate-all", index_metadata_freq: int = 15) -> None:
        self.text_splitter_options = text_splitter_options
        self.return_dict = return_dict
        self.cleaner = TranscriptCleaner(punctuation_model_name)
        self.index_metadata_freq = index_metadata_freq
    
    def _filter_index_metadata(self, entries):
        """
        Filters the index metadata to include only entries that occur after every n seconds.
        """
        if not entries:
            return []

        filtered_entries = []
        filtered_entries.append(entries[0])
        last_start_time = entries[0]['start_time']
        
        for entry in entries:
            if entry['start_time'] >= last_start_time + self.index_metadata_freq:
                filtered_entries.append(entry)
                last_start_time = entry['start_time']

        return filtered_entries
    
    def _clean_transcript(self, transcript_metadata: Dict, type: str, additional_metadata: Dict = {}) -> Document:
        if type == 'youtube':
            text, index = self.cleaner.clean_youtube_transcript(transcript_metadata['file_path'])
        elif type == 'echo360':
            text, index = self.cleaner.clean_echo360_transcript(transcript_metadata['file_path'])
        else:
            raise ValueError(f"Invalid transcript type: {type}")
        
        index = self._filter_index_metadata(index)
        additional_metadata.update({"index_metadata": index, 
                                    "video_title": transcript_metadata['title'], 
                                    "video_url": transcript_metadata['url'], 
                                    "video_desc": transcript_metadata['description']})
        transcript = {"content": text, "metadata": additional_metadata}
        return transcript
        
    def _clean_submodule_transcripts(self, submodule: Dict, additional_metadata: Dict = {}) -> List[Dict]:
        """Clean the transcripts of a submodule and add additional metadata."""
        cleaned_transcripts = []
        
        # Combine additional metadata with submodule-specific metadata
        submodule_metadata = {
            'subsection': submodule['subsection'],
            'submodule_title': submodule['submodule_title'],
            'submodule_url': submodule['submodule_url']
        }
        additional_metadata = {**additional_metadata, **submodule_metadata}

        for youtube_metadata in submodule['youtube_metadatas']:
            transcript = self._clean_transcript(youtube_metadata, "youtube", additional_metadata)
            cleaned_transcripts.append(transcript)

        for echo360_metadata in submodule['echo360_metadatas']:
            transcript = self._clean_transcript(echo360_metadata, "echo360", additional_metadata)
            cleaned_transcripts.append(transcript)

        return cleaned_transcripts

    def process_submodule_transcripts(self, submodule: Dict, additional_metadata: Dict = {}) -> List[Document]:
        """Process the transcripts of a submodule into a list of Documents."""
        cleaned_transcripts = self._clean_submodule_transcripts(submodule, additional_metadata)
        
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