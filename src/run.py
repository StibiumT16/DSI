import os
import json
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
    valid_callback : Optional[bool] = field(default=False)
    kmeans_path : Optional[str] = field(default=None) 
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


def load_encoded_docid(tokenizer, docid_path):
    docids = []
    max_len = 0
    print("Loading docid")
    with open(docid_path, "r") as fr:
        for line in tqdm(fr):
            id, cluster_ids = line.strip().split("\t")
            cluster_ids = cluster_ids.replace(',', ' ')
            docid = tokenizer(cluster_ids)['input_ids']
            if len(docid) > max_len:
                max_len = len(docid)
            docids.append(docid)
    return docids, max_len


def train(model, train_dataset, training_args, callbacks):       
    trainer = T5Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        callbacks=callbacks,
        data_collator=torch_default_data_collator,
    )
    
    trainer.train()
    trainer.save_model()


def inference(model, tokenizer, test_dataset, trie, max_docid_length, training_args, run_args):
    def _prefix_allowed_tokens_fn(batch_id, sent):
        return trie.get(sent.tolist())
    
    def docid2clusterid(docid):
        x_list = []
        for x in docid:
            if x == 1:
                break
            elif x != 0:
                x_list.append(str(x))
        res = tokenizer.decode(x_list).replace(' ', ',')
        return res

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
    set_seed(run_args.random_seed)
    training_args.seed=run_args.random_seed
    
    if run_args.task == 'train':
        tokenizer = T5Tokenizer.from_pretrained(run_args.model_name_or_path)
        model = T5ForConditionalGeneration.from_pretrained(run_args.model_name_or_path)
        docids, max_docid_length = load_encoded_docid(tokenizer, run_args.kmeans_path)
        max_docid_length += 1
        train_dataset = Traindata(run_args.train_path, run_args.max_length, max_docid_length, tokenizer)

        if run_args.valid_callback:
            trie = Trie([[0] + item for item in docids])
            valid_dataset = Testdata(run_args.valid_path, run_args.max_length, tokenizer)
            callbacks=[EvalCallback(valid_dataset, tokenizer, trie, max_docid_length+1, run_args.num_beams, training_args)]
        else:
            callbacks = None
        train(model, train_dataset, training_args, callbacks)
                
    elif run_args.task == 'inference':
        tokenizer = T5Tokenizer.from_pretrained(run_args.backbone_model)
        model = T5ForConditionalGeneration.from_pretrained(run_args.model_name_or_path).to(torch.device("cuda:0")).eval()
        docids, max_docid_length = load_encoded_docid(tokenizer, run_args.kmeans_path)
        max_docid_length += 1
        trie = Trie([[0] + item for item in docids])
        test_dataset = Testdata(run_args.test_path, run_args.max_length, tokenizer)
        inference(model, tokenizer, test_dataset, trie, max_docid_length, training_args, run_args)