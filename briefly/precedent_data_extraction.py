import re
import json
import bisect
import os
from datetime import date
from joblib import Parallel, delayed
from rapidfuzz import fuzz, process
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK models quietly if not already present
nltk.download('punkt', quiet=True)

tqdm.pandas()

############################################################################################################
# ----- Step 0: Get case data (from Hugging Face LePaRD) -----
############################################################################################################

def make_case_data_from_hf(n=None, split="train", streaming=False):
    """
    Loads LePaRD dataset directly from Hugging Face instead of local CAP data.
    Saves compatible JSON structure for downstream tokenization and processing.
    """
    print("Loading LePaRD dataset from Hugging Face...")
    dataset = load_dataset("rmahari/LePaRD", split=split, streaming=streaming)

    data = []
    for i, record in enumerate(dataset):
        if n and i >= n:
            break

        case_dict = {
            'id': record.get('dest_id', f"case_{i}"),
            'name': record.get('dest_name', f"Unknown_{i}"),
            'text': record.get('destination_context', ''),
            'cites_to': [{'cite': record.get('source_cite', '')}],
            'court': record.get('dest_court', ''),
            'date': record.get('dest_date', ''),
            'cite': record.get('dest_cite', '')
        }

        if not case_dict['text']:
            continue

        data.append(case_dict)
        if i % 1000 == 0:
            print(f"Processed {i} records", end='\r')

    meta = f"Created on {date.today()}. Derived from LePaRD dataset."
    cases = {'meta': meta, 'data': data}

    os.makedirs("./data", exist_ok=True)
    with open('./data/case_data.json', 'w') as fp:
        json.dump(cases, fp)

    print(f"\nSaved {len(data)} cases to ./data/case_data.json")

############################################################################################################
# ----- Step 1: Tokenize Cases (Lightweight NLTK Version) -----
############################################################################################################

def legal_sentence_tokenize(text):
    """
    Tokenizes text into sentences using lightweight NLTK tokenizer.
    """
    if not text:
        return []
    return sent_tokenize(text)

def tokenize_cases_nltk():
    """
    Tokenizes all cases using NLTK instead of a transformer model.
    """
    print("Tokenizing cases with NLTK...")

    with open('./data/case_data.json') as f:
        case_texts = json.load(f)
    case_texts = case_texts['data']

    for i, case in enumerate(tqdm(case_texts, desc="Tokenizing")):
        case['text'] = legal_sentence_tokenize(case['text'])
    
    os.makedirs("./data", exist_ok=True)
    with open('./data/case_data_tokenized.json', 'w') as f:
        json.dump({'meta': f'Tokenized with NLTK on {date.today()}', 'data': case_texts}, f)

    print(f"Saved tokenized data: {len(case_texts)} cases → ./data/case_data_tokenized.json")

############################################################################################################
# ----- Step 2: Citation Matcher ----- #
############################################################################################################

def citation_matcher(case, quoted_precedent):
    """
    Sub-Routine: Given a list of quoted precedent from precedent_extractor,
    match these to sentences in the corresponding origin case
    """
    case_id = case['id']
    case_text = case['text'] 
    choices = [x for x in case_text if len(x.split()) > 10]

    extracted = []
    score_cutoff = 90

    for info in quoted_precedent:
        dest_id = info['dest_id']
        
        for query in info['quoted_text']:
            match = process.extract(query['text'],
                                    choices,
                                    scorer=fuzz.token_set_ratio,
                                    score_cutoff=score_cutoff)
            
            if match: 
                match = [x for x in match if x[1] == max([x[1] for x in match])]
                match = match[0][0]  

                extracted.append({
                    'dest_id': dest_id, 
                    'source_id': case_id, 
                    'passage': match, 
                    'query': query,
                    'dest_date': info['dest_date'],
                    'dest_court': info['dest_court'],
                    'dest_name': info['dest_name'],
                    'dest_cite': info['dest_cite'],
                    'source_date': case['date'], 
                    'source_court': case['court'],
                    'source_name': case['name'],
                    'source_cite': case['cite']
                })
    
    if len(extracted) == 0: 
        return None
    return extracted

############################################################################################################
# ----- Step 3: Assign Passage IDs ----- #
############################################################################################################

def assign_passage_id(passages_origin_filename = './data/citations_origin.json', passage_dict_filename = './data/passage_dict.json'):
    """
    Assigns a unique passage_id to each unique passage and saves the mapping of passage_id to passage.
    """
    with open(passages_origin_filename) as f:
        citations_origin = json.load(f)['data']

    unique_passages = [{'source_id': x['source_id'], 'passage': x['passage']} for x in citations_origin]
    unique_passages = list(set(tuple(p.items()) for p in unique_passages))
    unique_passages = [dict(p) for p in unique_passages]

    print('Number of unique passages:', len(unique_passages))
    
    grouped_passages = defaultdict(list)
    for passage in unique_passages:
        source_id = passage['source_id']
        passage_id = f"{source_id}_{len(grouped_passages[source_id])}"
        grouped_passages[source_id].append({'passage_id': passage_id, 'passage': passage['passage']})

    grouped_passages_dict = {item['source_id']: item['passages'] for item in [{'source_id': k, 'passages': v} for k, v in grouped_passages.items()]}

    updated_citations_origin = []
    for citation in citations_origin:
        source_id = citation['source_id']
        passage = citation['passage']
        updated_citation = citation.copy()
        passage_id = None
        for p in grouped_passages_dict[source_id]:
            if p['passage'] == passage:
                passage_id = p['passage_id']
                break
        if passage_id is not None:
            updated_citation.pop('passage')
            updated_citation['passage_id'] = passage_id
        updated_citations_origin.append(updated_citation)

    os.makedirs("./data", exist_ok=True)
    with open('./data/citations_origin_updated.json', 'w') as fp:
        json.dump({'meta': f'Updated on {date.today()}', 'data': updated_citations_origin}, fp)

    flat_dict = {p['passage_id']: p['passage'] for value in grouped_passages_dict.values() for p in value}
    with open(passage_dict_filename, 'w') as fp:
        json.dump({'meta': f'Passage dictionary on {date.today()}', 'data': flat_dict}, fp)

############################################################################################################
# ----- Step 4: Context Wrapper (for completeness) ----- #
############################################################################################################

def context_wrapper(training_filename='./data/training_data.csv.gz', num_cores=4, batch_size=500):
    """
    Parallelizes get_destination_context (optional downstream use)
    """
    print("Context wrapper placeholder — this step depends on citation alignment output.")
    # If you build retrieval instead, you can skip this entirely.

############################################################################################################
# ----- Main Entrypoint ----- #
############################################################################################################

if __name__ == "__main__":
    # Step 0: Load and save data from Hugging Face
    # make_case_data_from_hf(n=5000)  # Uncomment for initial run

    # Step 1: Lightweight tokenization
    tokenize_cases_nltk()