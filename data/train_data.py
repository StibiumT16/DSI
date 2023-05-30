import json
import random
from tqdm import tqdm
from nltk.tokenize import word_tokenize

data = 'nq320k'
first_k_tokens = 64

corpus_path = f'{data}/dataset/corpus.json'
kmeans_path = f'{data}/kmeans/bert_10_100.tsv'
train_qrels_path = f'{data}/dataset/nq-doctrain-qrels.tsv'
train_queries_path = f'{data}/dataset/nq-doctrain-queries.tsv'
train_out_path = f'{data}/train_data/train.json'
dev_qrels_path = f'{data}/dataset/nq-docdev-qrels.tsv'
dev_queries_path = f'{data}/dataset/nq-docdev-queries.tsv'
dev_out_path = f'{data}/train_data/dev.json'
valid_out_path = f'{data}/train_data/valid.json'

def id2docid():
    id2docid = {}
    with open(kmeans_path, 'r') as fr:
        for line in tqdm(fr):
            id, docid = line.strip().split('\t')
            id2docid[id] = docid
    return id2docid

def get_query(path):
    qid2query = {}
    with open(path, 'r') as fr:
        for line in fr:
            qid, q = line.strip().lower().split('\t')
            qid2query[qid] = q
    return qid2query

def train_data():
    def first_tokens(text, k):
        return ' '.join(word_tokenize(text)[:k])
    
    id2clusterid = id2docid()
    qid2query = get_query(train_queries_path)
    with open(train_out_path, 'w') as fw:
        
        print("Document training data:")
        print(f"Strategy: First {first_k_tokens} tokens")
        with open(corpus_path, 'r') as fr:
            for line in tqdm(fr):
                line = json.loads(line)
                id, context = line['docid'], line['body'].strip().lower()
                text = first_tokens(context, first_k_tokens)
                json_str = json.dumps({'id' : id2clusterid[id], 'context' : text})
                fw.write(json_str + '\n')
        
        print("Query training data:")
        with open(train_qrels_path, 'r') as fr:
            for line in tqdm(fr):
                qid, _, did, _ = line.split('\t')
                json_str = json.dumps({'id' : id2clusterid[did], 'context' : qid2query[qid]})
                fw.write(json_str + '\n')
                
def dev_data():
    id2clusterid = id2docid()
    qid2query = get_query(dev_queries_path)
    with open(dev_out_path, 'w') as fw:
        with open(dev_qrels_path, 'r') as fr:
            for line in tqdm(fr):
                qid, _, did, _ = line.split('\t')
                json_str = json.dumps({'id' : id2clusterid[did], 'context' : qid2query[qid]})
                fw.write(json_str + '\n')

def sample_valid_data():
    with open(dev_out_path, 'r') as fr, open(valid_out_path, 'w') as fw:
        lines = [line for line in fr]
        random.shuffle(lines)
        for line in lines[:100]:
            fw.write(str(line))
        
        
train_data()
dev_data()
sample_valid_data()