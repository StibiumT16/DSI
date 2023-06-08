# url-title docid in Ultron

import re
import json
import random
import argparse 
from tqdm import tqdm
from transformers import AutoTokenizer 

parser = argparse.ArgumentParser()
parser.add_argument('--corpus_path', type=str, default='dataset_name/dataset/corpus.json')
parser.add_argument('--output_path', type=str)
parser.add_argument('--model_name_or_path', type=str, default='t5-base')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)


with open(args.corpus_path, 'r') as fr, open(args.output_path, 'w') as fw:
    for line in tqdm(fr):
        line = json.loads(line)
        old_id, url, title = line['docid'], line['url'].lower(), line['title']
        url = url.replace("http://","").replace("https://","").replace("-"," ").replace("_"," ").replace("?"," ").replace("="," ").replace("+"," ").replace(".html","").replace(".php","").replace(".aspx","").replace(".org","").replace(";"," ").strip()
        reversed_url = url.split('/')[::-1]
        url_content = " ".join(reversed_url[:-1])
        domain = reversed_url[-1]
            
        url_content = ''.join([i for i in url_content if not i.isdigit()])
        url_content = re.sub(' +', ' ', url_content).strip()
        
        if len(title.split()) <= 2:
            url = url_content + " " + domain
        else:
            url = title + " " + domain
        new_id = tokenizer(url).input_ids
        new_id = ",".join([str(x) for x in new_id])
        fw.write(old_id + '\t' + new_id + '\n')


        