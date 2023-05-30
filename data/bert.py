import json
import torch
import argparse
from tqdm import tqdm
from transformers import BertTokenizer, AutoModel

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", default="corpus.json", type=str)
parser.add_argument("--output_file", default="emb_bert.txt", type=str)
parser.add_argument("--max_length", default=512, type=int)
args = parser.parse_args()

device = torch.device("cuda:0")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained("bert-base-uncased").to(device)

with open(args.input_file, 'r') as fr, open(args.output_file, 'w') as fw:
    for line in tqdm(fr):
        line = json.loads(line)
        did, text = line['docid'], (line['title'] + ' ' + line['body']).lower()
        input = tokenizer(text, 
                          max_length=args.max_length, 
                          return_tensors='pt', 
                          padding=True, 
                          truncation=True
                        ).to(device)
        output = model(**input, return_dict=True).last_hidden_state.detach().cpu()[:, 0, :].numpy().tolist()[0]
        fw.write(str(did) + "\t" + ",".join([str(x) for x in output]) + '\n')
        