import json
import random
import argparse 
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--corpus_path', type=str, default='dataset_name/dataset/corpus.json')
parser.add_argument('--output_path', type=str)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)

old_ids, new_ids = [], []

with open(args.corpus_path, 'r') as fr:
    for i, line in enumerate(tqdm(fr)):
        line = json.loads(line)
        old_ids.append(line['docid'])
        new_ids.append(i)

random.shuffle(new_ids)

with open(args.output_path, 'w') as fw:
    for old_id, new_id in tqdm(zip(old_ids, new_ids)):
        fw.write(old_id + '\t' + str(new_id) + '\n')