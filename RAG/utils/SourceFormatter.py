import re
from typing import Dict, List, Optional


class SourceFormatter:
    def __init__(self, source_config: Optional[Dict] = None):
        self.source_config = source_config or {
            "video_transcript": {
                "url_key": "video_url",
                "title_key": "video_title",
                "block_id_key": "start_time",
            },
            "html_content": {
                "url_key": "submodule_url",
                "title_key": "submodule_title",
                "block_id_key": "data_block_id",
            },
        }
    @staticmethod
    def escape_square_brackets(text):
        """
        Escape all square brackets ([ and ]) in the given text.
        """
        text = re.sub(r'\[', r'\\[', text)  
        text = re.sub(r'\]', r'\\]', text)  
        return text

    # @staticmethod
    # def _generate_contextual_header(source_metadata: Dict) -> str:
    #     """
    #     Generates a contextual header based on the metadata of the source.
    #     """
    #     if source_metadata['content_type'] == 'video_transcript':
    #         # Replace newlines with a single space and truncate to 1000 characters
    #         video_desc = re.sub(r'\n+', ' ', source_metadata['video_desc'][:1000])
    #         return f"Below are transcript snippet from video with a description of: {video_desc.strip()}:\n"
        
    #     elif source_metadata['content_type'] == 'html_content':
    #         return (
    #             f"Below are content snippet of: {source_metadata['module_title']} - "
    #             f"{source_metadata['subsection']}: {source_metadata['submodule_title']}:\n"
    #         )
            
    @staticmethod
    def _merge_overlapping_sources(sources: List[Dict], url_key: str) -> List[Dict]:
        """
        Merges overlapping sources based on the URL key.
        """
        url_dict = {}

        # Group sources by URL
        for source in sources:
            source['metadata']['end_index'] = source['metadata']['start_index'] + len(source['text'])
            if source['metadata'][url_key] not in url_dict:
                url_dict[source['metadata'][url_key]] = []
            url_dict[source['metadata'][url_key]].append(source)

        # Merge overlapping sources for each URL
        for url, sources in url_dict.items():
            if len(sources) > 1:
                new_sources = []
                # Sort documents by start_index
                sorted_sources = sorted(sources, key=lambda x: x['metadata']['start_index'])
                for i, source in enumerate(sorted_sources):
                    if i == 0:
                        new_sources.append(source)
                    else:
                        # Check if the current source overlaps with the last source in new_sources
                        if source['metadata']['start_index'] < new_sources[-1]['metadata']['end_index']:
                            # Calculate the non-overlapping text
                            non_overlapping_text = source['text'][new_sources[-1]['metadata']['end_index'] - source['metadata']['start_index']:]
                            new_sources[-1]['text'] += non_overlapping_text
                            new_sources[-1]['metadata']['end_index'] = source['metadata']['end_index']
                        else:
                            new_sources.append(source)
                # Update the URL dict with the merged sources
                url_dict[url] = new_sources

        # Flatten the dictionary back to a list
        merged_sources = []
        for sources in url_dict.values():
            merged_sources.extend(sources)

        return merged_sources

    def _split_source_by_block(self, source: Dict, config: Dict) -> Dict:
        """
        Splits a source into blocks based on the configuration.
        """
        source_splits = []
        start_index = source['metadata']['start_index']
        end_index = start_index + len(source['text'])

        # Determine the block identifier key
        block_id_key = config["block_id_key"]

        sorted_blocks = source['index_metadata']
        for i, block in enumerate(sorted_blocks):
            block_id = block[block_id_key]
            block_start = block['char_start']
            # Determine the effective end of the block from the next block's start
            next_block_start = (
                sorted_blocks[i + 1]['char_start'] if i + 1 < len(sorted_blocks) else len(source['text'])
            )
            block_end = next_block_start

            # Check if there is overlap with the index range
            if start_index < block_end and end_index > block_start:
                # Calculate the slice range within the block
                adjust_start = max(0, block_start - start_index)
                adjusted_end = min(len(source['text']), block_end - start_index)
                text = source['text'][adjust_start:adjusted_end]

                source_splits.append(
                    {
                        "block_id": block_id,
                        "text": text,
                    }
                )

        # Create the source dictionary
        source_dict = {
            "url": source['metadata'][config["url_key"]],
            "title": source['metadata'][config["title_key"]],
            "contextual_header": source['metadata']['contextual_header'],
            "source_splits": source_splits,
            "content_type": source['metadata']['content_type'],
        }

        return source_dict
    
    def format_sources_for_llm(self, sources: List[Dict]) -> Dict:
        """
        Formats sources for LLM input, combining both video and HTML sources.
        """
        # Get configurations for both source types
        html_config = self.source_config["html_content"]
        video_config = self.source_config["video_transcript"]
        
        # Split sources into HTML sources and video
        html_sources = [source for source in sources if source["metadata"]['content_type']== "html_content"]
        video_sources = [source for source in sources if source["metadata"]['content_type']== "video_transcript"]
        
        # Merge overlapping sources for both HTML and video
        merged_html_sources = self._merge_overlapping_sources(html_sources, html_config["url_key"])
        merged_video_sources = self._merge_overlapping_sources(video_sources, video_config["url_key"])
        
        for source in merged_html_sources:
            source['config'] = html_config
        for source in merged_video_sources:
            source['config'] = video_config
            
        # Combine both source types into a single list
        combined_sources = merged_html_sources + merged_video_sources

        # Split sources into blocks and format them
        source_dicts = []
        source_id = 0
        final_formatted_sources = ""

        for source in combined_sources:
            config = source['config']

            # Split the source into blocks
            source_dict = self._split_source_by_block(source, config)
            formatted_splits = ""

            # Assign source IDs and format the splits
            for split in source_dict['source_splits']:
                split['source_id'] = source_id
                text = self.escape_square_brackets(split['text']).strip()
                formatted_splits += f"[{source_id}]\n{text}\n"
                source_id += 1

            # Add source IDs to the source dictionary
            source_dict['source_ids'] = [split['source_id'] for split in source_dict['source_splits']]
            source_dicts.append(source_dict)

            # Add the formatted splits to the final output
            final_formatted_sources += "===\n" + source_dict['contextual_header'] + "\n" + formatted_splits + "\n"

        return {"content": final_formatted_sources, "source_dicts": source_dicts}