import datetime
import re
from typing import Dict, List
from urllib.parse import urlparse, urlunparse


def _add_timestamp_to_url(url, start_time):
    """
    Add a timestamp query parameter to the video URL.
    """
    # Parse the URL to list because the ParseResult tuple is immutable
    url_parts = list(urlparse(url))

    # Add the timestamp query parameter
    query = f"t={int(start_time)}"
    
    # the query parameter is the 5th element in the list
    # If there's already a query string, append the new parameter
    if url_parts[4]:  
        url_parts[4] += "&" + query
    # Otherwise, create a new query string
    else:
        url_parts[4] = query

    # Reconstruct the URL
    return urlunparse(url_parts)

def _add_timestamps_to_title(title, block_ids, extra_end_time=15):
    """Modify the video title with timestamps. 
    block_ids is a sorted list of start timestamps in seconds.
    The extra_end_time should be adjusted based on index_metadata_freq arg on TranscriptDocProcessor.
    """
    start_time = int(block_ids[0])
    end_time = int(block_ids[-1]) + extra_end_time
    
    # convert start time and end time to hh:mm:ss format
    start_time = str(datetime.timedelta(seconds=start_time))
    end_time = str(datetime.timedelta(seconds=end_time))
    return f"{title} ({(start_time)}-{end_time})"
    
def process_video_citation(citation, merge_threshold=60):
    """
    Process video citation and combine block_ids if they are continuous within 30 seconds.
    """
    # Extract and sort block_ids
    block_ids = sorted(citation['block_ids'])
    combined_citation = []
    current_group = []
    earliest_time = None

    for block_id in block_ids:
        if not current_group:
            # Start a new group
            current_group.append(block_id)
            earliest_time = block_id
        else:
            # Check if the current block_id is within merge threshold (in seconds) of the last one in the group
            last_time = current_group[-1]
            if block_id - last_time <= merge_threshold:
                current_group.append(block_id)
            else:
                # End the current group and start a new one
                combined_citation.append({
                    'url': _add_timestamp_to_url(citation['url'], earliest_time),
                    'title': _add_timestamps_to_title(citation['title'], block_ids),
                    'old_citation_ids': citation['old_citation_ids'],
                    'block_ids': current_group,
                    'content_type': citation['content_type']
                })
                current_group = [block_id]
                earliest_time = block_id

    # Add the remaining last group
    if current_group:
        combined_citation.append({
            'final_url': _add_timestamp_to_url(citation['url'], earliest_time),
            'title': _add_timestamps_to_title(citation['title'], block_ids),
            'old_citation_ids': citation['old_citation_ids'],
            'block_ids': current_group,
            'content_type': citation['content_type']
        })

    return combined_citation

class CitationFormatter:
    @staticmethod
    def _deduplicate_consecutive_citations(text: str) -> str:
    # Find groups of consecutive citations and deduplicate them
        def replace_consecutive(match):
            citations = re.findall(r'\[(\d+)\]', match.group(0))
            unique_citations = []
            for citation in citations:
                # Only remove duplicates within this consecutive group
                if citation not in unique_citations:
                    unique_citations.append(citation)
            return ''.join(f'[{citation}]' for citation in unique_citations)
        
        # Pattern to match one or more consecutive citations
        pattern = r'(?:\[\d+\])+' 
        return re.sub(pattern, replace_consecutive, text)
    
    def format_final_answer(self, answer: str, source_metadata: List[Dict]) -> Dict:
        # extract the citation ids from the answer
        citation_pattern = re.compile(r"\^\[(\d+)\]")
        citation_ids = citation_pattern.findall(answer)
        citation_ids = set([int(id) for id in citation_ids])

        # find the source for each citation
        citation_data = []

        for source in source_metadata:
            matching_ids = set(source['source_ids']).intersection(citation_ids)
            if matching_ids:
                citation = {"url": source['url'], 
                            "title": source['title'],
                            "old_citation_ids": [], 
                            "block_ids": [],
                            "content_type": source['content_type']}
                for source_split in source['source_splits']:
                    if source_split['source_id'] in matching_ids:
                        citation['old_citation_ids'].append(source_split['source_id'])
                        citation['block_ids'].append(source_split['block_id'])
                if citation['content_type'] == 'video_transcript':
                    citation_data.extend(process_video_citation(citation))
                elif citation['content_type'] == 'html_content':
                    citation['final_url'] = citation['url'] + "/block/" + ",".join(citation['block_ids'])
                    citation_data.append(citation)
        
        # Sort by the minimum value in 'citation_ids'
        citation_data = sorted(citation_data, key=lambda x: min(x['old_citation_ids']))
        # process the final citation url and id
        new_citation_id = 1
        for citation in citation_data:
            citation['new_citation_id'] = new_citation_id
            new_citation_id += 1

        # reformat the answer with the new citation ids
        final_answer = answer
        for citation in citation_data:
            for old_citation_id in citation['old_citation_ids']:
                final_answer = final_answer.replace(f"^[{old_citation_id}]", f"[{citation['new_citation_id']}]")
                
        final_citation = {}
        for citation in citation_data:
            final_citation[citation['new_citation_id']] = {"url": citation['final_url'], "title": citation['title']}
        
        # deduplicate the final citation
        final_answer = self._deduplicate_consecutive_citations(final_answer)
        
        return {"content": final_answer, "citation": final_citation, "citation_data": citation_data}