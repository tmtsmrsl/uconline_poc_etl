import re
from typing import Dict


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
    
    def format_final_answer(self, result: Dict) -> Dict:
        # extract the citation ids from the answer
        citation_pattern = re.compile(r"\^\[(\d+)\]")
        citation_ids = citation_pattern.findall(result['answer'])
        citation_ids = set([int(id) for id in citation_ids])

        # find the source for each citation
        citation_data = []
        source_metadata = result['formatted_sources']['source_dicts']

        for submodule_source in source_metadata:
            matching_ids = set(submodule_source['source_ids']).intersection(citation_ids)
            if matching_ids:
                citation = {"submodule_url": submodule_source['submodule_url'], 
                            "submodule_title": submodule_source['submodule_title'],
                            "old_citation_ids": [], 
                            "block_ids": []}
                for source_split in submodule_source['source_splits']:
                    if source_split['source_id'] in matching_ids:
                        citation['old_citation_ids'].append(source_split['source_id'])
                        citation['block_ids'].append(source_split['block_id'])
                citation_data.append(citation)

        # Sort by the minimum value in 'citation_ids'
        citation_data = sorted(citation_data, key=lambda x: min(x['old_citation_ids']))

        # process the final citation url and id
        new_citation_id = 1
        for citation in citation_data:
            citation['final_url'] = citation['submodule_url'] + "/block/" + ",".join(citation['block_ids'])
            citation['new_citation_id'] = new_citation_id
            new_citation_id += 1

        # reformat the answer with the new citation ids
        final_answer = result['answer']
        for citation in citation_data:
            for old_citation_id in citation['old_citation_ids']:
                final_answer = final_answer.replace(f"^[{old_citation_id}]", f"[{citation['new_citation_id']}]")
                
        final_citation = {}
        for citation in citation_data:
            final_citation[citation['new_citation_id']] = {"url": citation['final_url'], "title": citation['submodule_title']}
        
        # deduplicate the final citation
        final_answer = self._deduplicate_consecutive_citations(final_answer)
        
        return {"content": final_answer, "citation": final_citation}