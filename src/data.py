import torch
import datasets
import numpy as np
from typing import Dict, List
from torch.utils.data import Dataset

class Traindata(Dataset):
    def __init__(self, 
                 filename, 
                 max_length,
                 max_docid_length, 
                 tokenizer, 
        ):
        self.max_length = max_length
        self.max_docid_length = max_docid_length
        self.tokenizer = tokenizer
        self.data = datasets.load_dataset(
            'json',
            data_files = filename,
            ignore_verifications=False,
            cache_dir='cache',
        )['train']
        self.total_len = len(self.data)  
      
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, item):
        data = self.data[item]
        text, clusterid = data['context'], data['id']
        input = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
        )
        
        docid = self.tokenizer(clusterid.replace(',', ' '), 
                                truncation=True,
                                padding='max_length',
                                max_length=self.max_docid_length,
                                return_tensors="pt"
        ).input_ids[0]
        
        return {
            "input_ids": input.input_ids[0],
            "attention_mask": input.attention_mask[0],
            "labels": docid,
        }

class Testdata(Dataset):
    def __init__(self, 
                 filename, 
                 max_length, 
                 tokenizer, 
        ):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.data = datasets.load_dataset(
            'json',
            data_files = filename,
            ignore_verifications=False,
            cache_dir='cache',
        )['train']
        self.total_len = len(self.data)  
      
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, item):
        data = self.data[item]
        text, clusterid = data['context'], data['id']
        input = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
        )
        return {
            "input_ids": input.input_ids[0],
            "attention_mask": input.attention_mask[0],
            "labels": clusterid,
        }

class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)