import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from collections.abc import Mapping
from torch.utils.data import DataLoader, Dataset
from transformers.trainer import Trainer
from transformers.trainer_utils import is_main_process
from transformers import TrainingArguments, TrainerCallback
from typing import Dict, List, Tuple, Optional, Any, Union

class T5Trainer(Trainer):
    def __init__(self, **kwds):
        super().__init__(**kwds)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model.forward(input_ids=inputs["input_ids"],
                             attention_mask=inputs["attention_mask"], 
                             labels=inputs['labels'],
            ).loss
        if return_outputs:
            return loss, [None, None]
        return loss


class EvalCallback(TrainerCallback):
    def __init__(self, 
                 test_dataset, 
                 docid_strategy,
                 tokenizer,
                 model_vocab_size, 
                 trie, 
                 max_length : int, 
                 num_beams : int, 
                 args: TrainingArguments
                ):
        
        self.args = args
        self.trie = trie
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_beams = num_beams
        self.bias = model_vocab_size
        self.docid_strategy = docid_strategy
        self.dataloader = DataLoader(
            test_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
        )
        
    def _prefix_allowed_tokens_fn(self, batch_id, sent):
        return self.trie.get(sent.tolist())
    
    def cal_hit(self, truth, pred):
        hit1, hit10 = 0, 0
        for did, lst in zip(truth, pred):
            if did in lst[:1]:
                hit1 += 1
            if did in lst[:10]:
                hit10 += 1
        return hit1 / len(truth), hit10 / len(truth) 
    
    def docid2clusterid(self, docid):
        x_list = []
        if self.docid_strategy == 'expand_vocab':
            for x in docid:
                if x == 1:
                    break
                elif x >= self.bias:
                    x_list.append(str(x - self.bias))
        elif self.docid_strategy == 'tokenize':
            for x in docid:
                if x == 1:
                    break
                elif x != 0:
                    x_list.append(x)
            x_list = self.tokenizer.decode(x_list).split()
        elif self.docid_strategy == 'token':
            for x in docid:
                if x != 0:
                    x_list.append(str(x))
                elif x == 1:
                    break
        return ",".join(x_list)
    
    def on_save(self, args, state, control, **kwargs): # DEBUG: on_epoch_begin
        if is_main_process(self.args.local_rank):
            model = kwargs['model'].eval()
            truth, prediction = [], []
            
            for testing_data in tqdm(self.dataloader, desc='Evaluating valid set'):
                input_ids = testing_data["input_ids"]
                attention_mask = testing_data["attention_mask"]
                labels = testing_data["labels"]
                truth.extend(labels)
                
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids.to(model.device),
                        attention_mask=attention_mask.to(model.device),
                        max_length=self.max_length,
                        num_beams=self.num_beams,
                        num_return_sequences=self.num_beams,
                        prefix_allowed_tokens_fn=self._prefix_allowed_tokens_fn,
                        do_sample=False,
                    )
                    
                    for j in range(input_ids.shape[0]):
                        batch_output = outputs[j*self.num_beams:(j+1)*self.num_beams].cpu().numpy().tolist()
                        doc_rank = [self.docid2clusterid(docid) for docid in batch_output]
                        prediction.append(doc_rank)
            
            hit1, hit10 = self.cal_hit(truth, prediction)
            print(f"hit@1:{hit1}, hit@10:{hit10}")


def torch_default_data_collator(features) -> Dict[str, Any]:

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor(np.array([f[k] for f in features]))

    return batch