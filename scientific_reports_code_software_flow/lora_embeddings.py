###If you’re memory-bound, do QLoRA (4-bit loading with bitsandbytes) and train the LoRA adapters with TRL’s SFTTrainer. 
import datetime
import csv
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, average_precision_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
import numpy as np
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
from peft import PeftModel
MAX_LEN = 4096

def get_embeddings(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(device)
    ###inputs = {k: v.to(model.device) for k, v in inputs.items()}
    ###forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
    ###final layer hidden states: shape (1, seq_len, hidden_dim)
    ###print("len(hidden_states)", len(hidden_states))
    last_layer = hidden_states[-1]
    ###The attention_mask is 0 if its padding, 1 if a real token. Need to get the last non-zero token
    mask = inputs["attention_mask"][0]
    last_index = mask.nonzero()[-1].item()
    last_embedding = last_layer[0, last_index, :]  # shape: [hidden_dim]
    ###The line below could return a padding layer
    ###last_embedding = last_layer[0, -1, :]
    return last_embedding.cpu().numpy()

if __name__ == "__main__":
    ###all prints have flush=True to make sure they are output from the Falcon cluster immediately
    print(datetime.datetime.now(), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device", flush=True)
    
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
    model.config.output_hidden_states = True
    ###model = PeftModel.from_pretrained(model,"lora_fraud_error_falcon_1gpu_8batch_5epochs",device_map="auto")
    model = PeftModel.from_pretrained(model,"lora_adapter_falcon_1gpu_8batch_5epochs",device_map="auto")
    model.eval()
    
    ###with open('retraction_watch_abstract30_early.csv', 'r', encoding="utf8") as file, open('retraction_watch_embeddings30_lora_fraud_error_8batch_5epochs_early.csv', 'w') as f:
    ###with open('retraction_watch_abstract30_early.csv', 'r', encoding="utf8") as file, open('retraction_watch_embeddings30_lorabatch8_early.csv', 'w') as f:
    with open('retraction_watch_abstract30_early.csv', 'r', encoding="utf8") as file, open('retraction_watch_embeddings30_lora_batch8_epochs5_early.csv', 'w') as f:
        writer = csv.writer(f)
        for i, line in enumerate(file):
            if i % 1000 == 0:
                print(i,datetime.datetime.now())
            ###print(i,end=".")
            if len(line) > 1:
                abstract = line[line.find('Abstract')+9:]
                abstract = abstract.replace('\n','').replace('"','')
                embedding = get_embeddings(abstract, model, tokenizer, device)
                writer.writerow([embedding[i] for i in range(len(embedding))])
                
    ###with open('retraction_watch_abstract30_recent.csv', 'r', encoding="utf8") as file, open('retraction_watch_embeddings30_lora_fraud_error_8batch_5epochs_recent.csv', 'w') as f:
    ###with open('retraction_watch_abstract30_recent.csv', 'r', encoding="utf8") as file, open('retraction_watch_embeddings30_lorabatch8_recent.csv', 'w') as f:
    with open('retraction_watch_abstract30_recent.csv', 'r', encoding="utf8") as file, open('retraction_watch_embeddings30_lora_batch8_epochs5_recent.csv', 'w') as f:
        writer = csv.writer(f)
        for i, line in enumerate(file):
            if i % 1000 == 0:
                print(i,datetime.datetime.now())
            ###print(i,end=".")
            if len(line) > 1:
                abstract = line[line.find('Abstract')+9:]
                abstract = abstract.replace('\n','').replace('"','')
                embedding = get_embeddings(abstract, model, tokenizer, device)
                writer.writerow([embedding[i] for i in range(len(embedding))])
     
    ###with open('retraction_watch_abstract30_early.csv', 'r', encoding="utf8") as file, open('semantic_scholar_embeddings30_lora_fraud_error_8batch_5epochs_early.csv', 'w') as f:
    ###with open('semantic_scholar_abstract30_early.csv', 'r', encoding="utf8") as file, open('semantic_scholar_embeddings30_lora_batch8_epochs5_early.csv', 'w') as f:
    with open('semantic_scholar_abstract30_early.csv', 'r', encoding="utf8") as file, open('semantic_scholar_embeddings30_lora_batch8_epochs5_early.csv', 'w') as f:
        writer = csv.writer(f)
        for i, line in enumerate(file):
            if i % 1000 == 0:
                print(i,datetime.datetime.now())
            ###print(i,end=".")
            if len(line) > 0:
                abstract = line.replace('\n','').replace('"','')
                embedding = get_embeddings(abstract, model, tokenizer, device)
                writer.writerow([embedding[i] for i in range(len(embedding))])
                
    ###with open('retraction_watch_abstract30_recent.csv', 'r', encoding="utf8") as file, open('semantic_scholar_embeddings30_lora_fraud_error_8batch_5epochs_recent.csv', 'w') as f:
    ###with open('semantic_scholar_abstract30_recent.csv', 'r', encoding="utf8") as file, open('semantic_scholar_embeddings30_lora_batch8_epochs5_recent.csv', 'w') as f:
    with open('semantic_scholar_abstract30_recent.csv', 'r', encoding="utf8") as file, open('semantic_scholar_embeddings30_lora_batch8_epochs5_recent.csv', 'w') as f:
        writer = csv.writer(f)
        for i, line in enumerate(file):
            if i % 1000 == 0:
                print(i,datetime.datetime.now())
            ###print(i,end=".")
            if len(line) > 0:
                abstract = line.replace('\n','').replace('"','')
                embedding = get_embeddings(abstract, model, tokenizer, device)
                writer.writerow([embedding[i] for i in range(len(embedding))])
                
    print("\n",datetime.datetime.now(), flush=True)