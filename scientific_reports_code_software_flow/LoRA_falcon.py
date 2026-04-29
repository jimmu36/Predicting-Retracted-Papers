###If you’re memory-bound, do QLoRA (4-bit loading with bitsandbytes) and train the LoRA adapters with TRL’s SFTTrainer. That’s exactly what the 
###HF Llama-3.1 blogs/tutorials demonstrate. Hugging Face
import numpy as np
import copy
from sklearn.metrics import roc_auc_score,average_precision_score
import datetime
from sklearn.metrics import f1_score, recall_score, precision_score, mean_squared_error, confusion_matrix, roc_auc_score, roc_curve, auc, average_precision_score
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, BitsAndBytesConfig
from torch.optim import AdamW 
from datasets import Dataset
import os
try:
    from accelerate.utils import memory as accel_mem
    # If accelerate is old and doesn't have clear_device_cache, add a no-op
    if not hasattr(accel_mem, "clear_device_cache"):
        def clear_device_cache():
            # Fallback for older accelerate: do nothing
            return
        accel_mem.clear_device_cache = clear_device_cache
except Exception:
    # If accelerate isn't installed or something else goes wrong,
    # we just skip the hack and let the normal error surface.
    pass
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
MAX_LEN = 4096
BATCH_SIZE = 8

###this class holds the information for each abstract
class Node:
    def __init__(self):
        self.abstract = []
        self.retracted_or_valid = -1   ###retracted = 1, not-retracted = 0
    def __repr__(self):
        return f'Node({self.error_or_fraud}, {self.abstract})'
 
###This class stores the parameters used to define the LoRA matrices and the training they will get in fine tuning Mistral.
class Lora_dataset(Dataset):
    ###def __init__(self, texts, labels, tokenizer, max_length=4096):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        ###If the index passed in is a list of indices (i.e. the batch size is > 1)
        if isinstance(idx, list):
            batch_texts = [self.texts[i] for i in idx]
            batch_labels = [self.labels[i] for i in idx]
            enc = self.tokenizer(batch_texts,truncation=True,padding="max_length",max_length=self.max_length,return_tensors="pt")
            return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": torch.tensor(batch_labels, dtype=torch.long)}
        ###Since the return is above, this is just the "else", which is for an integer index (batch size = 1)
        text = self.texts[idx]
        enc = self.tokenizer(text,truncation=True,max_length=self.max_length,return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),"attention_mask": enc["attention_mask"].squeeze(0),"labels": torch.tensor(self.labels[idx], dtype=torch.long)}

if __name__ == "__main__":
    ###all prints have flush=True to make sure they are output from the Falcon cluster immediately
    print(datetime.datetime.now(), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device", flush=True)
    
    recent_papers = []
    with open('retraction_watch_abstract30_recent.csv', 'r', encoding="utf8") as file:
        for i, line in enumerate(file):
            if i % 1000 == 0:
                print(i,datetime.datetime.now(), flush=True)
            if len(line) > 1:
                paper = Node()
                abstract = line[line.find('Abstract')+9:]
                abstract = abstract.replace('\n','').replace('"','')
                paper.abstract = abstract
                paper.retracted_or_valid = 1   ###1 is a retracted paper
                recent_papers.append(paper)
    with open('semantic_scholar_abstract30_recent.csv', 'r', encoding="utf8") as file:
        for i, line in enumerate(file):
            if i % 1000 == 0:
                print(i,datetime.datetime.now(), flush=True)
            if len(line) > 0:
                paper = Node()
                abstract = line.replace('\n','').replace('"','')
                paper.abstract = abstract
                paper.retracted_or_valid = 0   ###0 is a valid paper
                recent_papers.append(paper)

    early_papers = []
    with open('retraction_watch_abstract30_early.csv', 'r', encoding="utf8") as file:
        for i, line in enumerate(file):
            if i % 1000 == 0:
                print(i,datetime.datetime.now(), flush=True)
            ###print(i,end=".")
            if len(line) > 1:
                paper = Node()
                abstract = line[line.find('Abstract')+9:]
                abstract = abstract.replace('\n','').replace('"','')
                paper.abstract = abstract
                paper.retracted_or_valid = 1   ###1 is a retracted paper
                early_papers.append(paper)
    with open('semantic_scholar_abstract30_early.csv', 'r', encoding="utf8") as file:
        for i, line in enumerate(file):
            if i % 1000 == 0:
                print(i,datetime.datetime.now(), flush=True)
            ###print(i,end=".")
            if len(line) > 0:
                paper = Node()
                abstract = line.replace('\n','').replace('"','')
                paper.abstract = abstract
                paper.retracted_or_valid = 0   ###0 is a valid paper
                early_papers.append(paper)
                
    print("\nlen early papers, recent papers", len(early_papers), len(recent_papers), flush=True)
    
    ###Since the falcon cluster cannot access the internet, need to use a local copy of Mistral
    snapshot_dir = "/s/chromatin/e/nobackup/jim/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da"
    ###This sets the LoRA parameters to be 4bit quantization, QLoRA. Floating point 32 or 16, ore even 8bit quantization resulted in out of memory
    ###issues with CS fish machines. Using this quantization on the falcon cluster enabled going to batch sizes of 8 without running out of memory.
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir,use_fast=True,local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    ###debug statements to make sure the Mistral model was found
    ###print("HF_HOME in job:", os.environ.get("HF_HOME"), flush=True)
    ###print("HF_HUB_OFFLINE:", os.environ.get("HF_HUB_OFFLINE"), flush=True)
    ###model = AutoModelForSequenceClassification.from_pretrained("mistralai/Mistral-7B-v0.1",num_labels=2,quantization_config=bnb_config,device_map="auto",local_files_only=True)
    ###Num labels = 2 indicates binary classification, local files only makes sure that the local copy identified above is used, not one retrieved from the internet
    model = AutoModelForSequenceClassification.from_pretrained(snapshot_dir,num_labels=2,quantization_config=bnb_config,device_map="auto",local_files_only=True)
    print("Model loaded!", flush=True)
    ###Mistral does not have a pad token defined, it is needed for LoRA, so this identifies one
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)
    ###r is the size of the LoRA rank, 8 is a common middle value from the range 4, 8, 16, where bigger is more storage/compute. lora_alpha is the scaling of how
    ###much the LoRA matrix adapters influence the output, and since the influence is divided by r, this factor of 2 gives the LoRA matrix adapters a bit more
    ###influence on the output. q_proj and v_proj are the query and value projections in multi-headed attention. Applying LoRA only to these two significantly 
    ###reduces training time and usually captures most of the model’s tunability.
    config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05) 
    ###inject LoRA into the model 
    model = get_peft_model(model, config)
    ###With gradient checkpointing, PyTorch does NOT store many of the intermediate activations. During backprop, it recomputes the missing forward-pass activations 
    ###on the fly. This is done to save GPU memory for large models (like Mistral) at the expense of taking longer to do the recompute.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    ###this retrieves the device the model is currently on, should be GPU for this function
    device = next(model.parameters()).device
    ###This creates a data collator that takes a batch of tokenized examples and pads them so they all have the same length before feeding them into the model.
    ###This is useful because each batch will have a given "longest" example. This allows lower memory usage as all vectors in the batch are padded to the longest
    ###in that batch, not to the longest possible (MAX_LEN).
    collator = DataCollatorWithPadding(tokenizer,padding="longest",max_length=MAX_LEN,return_tensors="pt")
    train_dataset = Lora_dataset([early_papers[i].abstract for i in range(len(early_papers))], [early_papers[i].retracted_or_valid for i in range(len(early_papers))],
                                 tokenizer, max_length=MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collator)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-5)
    ###optimizer = AdamW(model.parameters(), lr=2e-5)
    print("len(train_loader)", len(train_loader), flush=True)
    '''
    np.random.seed(17)
    ###The recent_papers are paired, with the 1st half being retracted and the 2nd half being valid. These need to be pulled in pairs from the recent papers so 
    ###there is no bias introduced. A random selection of 10% of the 1st half are generated, and their corresponding 2nd half are added to these for the
    ###validation set. Using dataloader and Lora_dataset so that padding is taken care of, hard to do with loading all inputs at once.
    first_half_indices = np.random.choice(len(recent_papers)//2, size=0.1*len(recent_papers)//2, replace=False)
    indices = copy.deepcopy(first_half_indices)
    [indices.append(idx+len(recent_papers)//2) for idx in first_half_indices]
    val_dataset = Lora_dataset([recent_papers[i].abstract for i in indices], [recent_papers[i].retracted_or_valid for i in indices], tokenizer, max_length=MAX_LEN)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, collate_fn=collator)
    ###X_val = [recent_papers[i].abstract for i in indices]
    ###Y_val = [recent_papers[i].retracted_or_valid for i in indices]
    '''
    for epoch in range(5):
        model.train()
        total_loss = 0.0
        batches = 0
        for i, batch in enumerate(train_loader):
            if i % 100 == 0:
                print("i =", i, datetime.datetime.now(), flush=True)
            optimizer.zero_grad()
            ###batch = {k: v.to("cuda") for k, v in batch.items()}
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],labels=batch["labels"])
            loss = outputs.loss
            '''
            print("loss.requires_grad:", loss.requires_grad)
            print("any parameter requires_grad:",
                  any(p.requires_grad for p in model.parameters()))
            model.print_trainable_parameters()
            '''
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        avg_loss = total_loss/batches
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}", flush=True)
        '''
        model.eval()
        y_pred_probs = []
        Y_val = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                y_pred_logits = model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"]).logits
                y_pred_prob = torch.softmax(y_pred_logits, dim=-1)[:, 1].cpu().numpy()
                ###.extend() appends each element of the array to the list
                y_pred_probs.extend(y_pred_prob)
                Y_val.extend(batch["label"].cpu().numpy())
        aucpr = average_precision_score(Y_val, y_pred_probs)
        print(f"AUCPR: {aucpr:.4f}", flush=True)
        '''
        model.save_pretrained("lora_adapter_falcon_1gpu_8_5epochs")
    
    print("\n",datetime.datetime.now(), flush=True)