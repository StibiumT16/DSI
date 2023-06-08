import os
import json
import math
import torch
import random
from data import *
from T5Trainer import *
from dataclasses import dataclass, field
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, HfArgumentParser

@dataclass
class RunArguments:
    task : str = field(default=None)
    random_seed : Optional[int] = field(default=0)
    num_beams : Optional[int] = field(default=10)
    max_length : Optional[int] = field(default=128)
    docid_strategy : Optional[str] = field(default="token") 
    valid_callback : Optional[bool] = field(default=False)
    docid_path : Optional[str] = field(default=None) 
    train_path : Optional[str] = field(default=None) 
    valid_path : Optional[str] = field(default=None) 
    test_path : Optional[str] = field(default=None) 
    output_run_path : Optional[str] = field(default='run.json') 
    model_name_or_path : Optional[str] = field(default='t5-base') 
    backbone_model : Optional[str] = field(default='t5-base') 
 
   
def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_encoded_docid(strategy, docid_path, model_vocab_size = 32128, tokenizer = None):
    docids = []
    max_len, max_val = 0, 0
    print(f"Loading docid ({strategy}): ")
    with open(docid_path, "r") as fr:
        if strategy == 'expand_vocab':
            for line in tqdm(fr):
                id, cluster_ids = line.strip().split("\t")
                docid =  [int(x) + model_vocab_size for x in cluster_ids.split(',')]
                max_val = max(max_val, max(docid))
                docid.append(1)
                max_len = max(max_len, len(docid))
                docids.append(docid)
        elif strategy == 'tokenize':
            for line in tqdm(fr):
                id, cluster_ids = line.strip().split("\t")
                cluster_ids = cluster_ids.replace(',', ' ')
                docid = tokenizer(cluster_ids)['input_ids']
                max_len = max(max_len, len(docid))
                docids.append(docid)
        elif strategy == 'token':
            for line in tqdm(fr):
                id, cluster_ids = line.strip().split("\t")
                docid = [int(x) for x in cluster_ids.split(',')]
                max_len = max(max_len, len(docid))
                docids.append(docid)
    return docids, max_len, max_val


def train(model, tokenizer, train_dataset, training_args, callbacks):       
    trainer = T5Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        callbacks=callbacks,
        data_collator=torch_default_data_collator,
    )
    
    trainer.train()
    trainer.save_model()


def inference(model, tokenizer, model_vocab_size, test_dataset, trie, max_docid_length, training_args, run_args):
    def _prefix_allowed_tokens_fn(batch_id, sent):
        return trie.get(sent.tolist())
    
    def docid2clusterid(docid):
        x_list = []
        if run_args.docid_strategy == 'expand_vocab':
            for x in docid:
                if x == 1:
                    break
                elif x >= model_vocab_size:
                    x_list.append(str(x - model_vocab_size))
        elif run_args.docid_strategy == 'tokenize':
            for x in docid:
                if x == 1:
                    break
                elif x != 0:
                    x_list.append(x)
            x_list = tokenizer.decode(x_list).split()
        elif run_args.docid_strategy == 'token':
            for x in docid:
                if x != 0:
                    x_list.append(str(x))
                elif x == 1:
                    break
        return ",".join(x_list)
        
    dataloader = DataLoader(
            test_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            num_workers=training_args.dataloader_num_workers,
            shuffle=False,
            drop_last=False,
        )
    
    truth, preds = [], []
    k = run_args.num_beams
    
    for testing_data in tqdm(dataloader, desc='Evaluating dev set'):
        input_ids = testing_data["input_ids"]
        attention_mask = testing_data["attention_mask"]
        truth.extend(testing_data["labels"])
        with torch.no_grad():
            outputs = model.generate( #constrained beam search
                input_ids=input_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),
                max_length=max_docid_length,
                num_beams=k,
                num_return_sequences=k,
                prefix_allowed_tokens_fn=_prefix_allowed_tokens_fn,
                do_sample=False,)
            
            for j in range(input_ids.shape[0]):
                batch_output = outputs[j*k:(j+1)*k].cpu().numpy().tolist()
                doc_rank = [docid2clusterid(docid) for docid in batch_output]
                preds.append(doc_rank)
            
    with open(run_args.output_run_path, 'w') as fw:
        for ans, pred in zip(truth, preds):
            fw.write(json.dumps({'truth' : ans, 'pred' : pred}) + "\n") 
            
         
if __name__ == '__main__':
    parser = HfArgumentParser((TrainingArguments, RunArguments))
    training_args, run_args = parser.parse_args_into_dataclasses()
    assert run_args.docid_strategy in ['tokenize', 'expand_vocab', 'token']
    set_seed(run_args.random_seed)
    training_args.seed=run_args.random_seed
    
    tokenizer = T5Tokenizer.from_pretrained(run_args.model_name_or_path)
    
    if run_args.task == 'train':
        model = T5ForConditionalGeneration.from_pretrained(run_args.model_name_or_path)
        model_vocab_size = model.config.vocab_size # 32128 for t5-base
        docids, max_docid_length, max_val = load_encoded_docid(run_args.docid_strategy, run_args.docid_path, model_vocab_size, tokenizer)
        
        if run_args.docid_strategy == 'expand_vocab':
            model.resize_token_embeddings(max_val + 1)
        train_dataset = Traindata(run_args.train_path, run_args.max_length, max_docid_length, tokenizer, model_vocab_size, run_args.docid_strategy)

        if run_args.valid_callback:
            trie = Trie([[0] + item for item in docids])
            valid_dataset = Testdata(run_args.valid_path, run_args.max_length, tokenizer)
            callbacks=[EvalCallback(valid_dataset, run_args.docid_strategy, tokenizer, model_vocab_size, trie, max_docid_length + 1, run_args.num_beams, training_args)]
        else:
            callbacks = None
            
        train(model, tokenizer, train_dataset, training_args, callbacks)

                
    elif run_args.task == 'inference':
        model_vocab_size = T5ForConditionalGeneration.from_pretrained(run_args.backbone_model).config.vocab_size # 32128 for t5-base
        model = T5ForConditionalGeneration.from_pretrained(run_args.model_name_or_path).to(torch.device("cuda:0")).eval()
        docids, max_docid_length, max_val = load_encoded_docid(run_args.docid_strategy, run_args.docid_path, model_vocab_size, tokenizer)
        trie = Trie([[0] + item for item in docids])
        test_dataset = Testdata(run_args.test_path, run_args.max_length, tokenizer)
        inference(model, tokenizer, model_vocab_size, test_dataset, trie, max_docid_length + 1, training_args, run_args)
