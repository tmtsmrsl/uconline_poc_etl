from typing import Dict, List


class SourceFormatter:
    @staticmethod
    def _merge_overlapping_sources(sources: List[Dict]) -> List[Dict]:
        submodule_url_dict = {}

        # Group sources by submodule_url
        for source in sources:
            source['end_index'] = source['start_index'] + len(source['text'])
            if source['submodule_url'] not in submodule_url_dict:
                submodule_url_dict[source['submodule_url']] = []
            submodule_url_dict[source['submodule_url']].append(source)

        # Merge overlapping sources for each submodule_url
        for submodule_url, sources in submodule_url_dict.items():
            if len(sources) > 1:
                new_sources = []
                # Sort documents by start_index
                sorted_sources = sorted(sources, key=lambda x: x['start_index'])
                for i, source in enumerate(sorted_sources):
                    if i == 0:
                        new_sources.append(source)
                    else:
                        # Check if the current source overlaps with the last source in new_sources
                        if source['start_index'] < new_sources[-1]['end_index']:
                            # note that we don't use the end_index of the last source directly as the indices are relative to the original submodule content
                            non_overlapping_text = source['text'][new_sources[-1]['end_index'] - source['start_index']:]
                            new_sources[-1]['text'] += non_overlapping_text
                            new_sources[-1]['end_index'] = source['end_index']
                        else:
                            new_sources.append(source)
                # Update the submodule_url_dict with the merged sources
                submodule_url_dict[submodule_url] = new_sources
                
        # Flatten the dictionary back to a list
        merged_sources = []
        for sources in submodule_url_dict.values():
            merged_sources.extend(sources)
            
        return merged_sources

    @staticmethod
    def _split_source_by_block(source: Dict) -> List[Dict]:
        source_splits = []
        start_index = source['start_index']
        end_index = start_index + len(source['text'])
        contextual_header = f"Below are content snippet of: {source['module_title']} - {source['subsection']}: {source['submodule_title']}"
                
        for block_id, block_range in source['data_block_ranges'].items():
            if start_index <= block_range['char_end'] and end_index >= block_range['char_start']:
                adjust_start = max(0, block_range['char_start']-start_index)
                adjusted_end = min(len(source['text']), block_range['char_end']-start_index)
                text = source['text'][adjust_start:adjusted_end]
                source_splits.append(
                    {   
                        "block_id": block_id,
                        "text": text,
                    }
                )
        
        source_dict = {
            "submodule_url": source['submodule_url'],
            "submodule_title": source['submodule_title'],
            "contextual_header": contextual_header,
            "source_splits": source_splits
        }
        
        return source_dict

    def format_sources_for_llm(self, sources: List[Dict]) -> Dict:
        merged_sources = self._merge_overlapping_sources(sources)
        source_dicts = []
        source_id = 0
        final_formatted_sources = ""
        
        for source in merged_sources:
            source_dict = self._split_source_by_block(source)
            formatted_splits = ""
            for split in source_dict['source_splits']:
                split['source_id'] = source_id
                formatted_splits += f"[{split['source_id']}]\n{split['text'].strip()}\n"
                source_id += 1
            source_dict['source_ids'] = [split['source_id'] for split in source_dict['source_splits']]
            source_dicts.append(source_dict)
            final_formatted_sources += source_dict['contextual_header'] + ":\n" + formatted_splits + "===\n\n"
            
        return {"content": final_formatted_sources, "source_dicts": source_dicts}