####mistral31_word_shap4 takes a file of 1000 previously identified strongly predicted retracted papers (>75%) and calculates the shap values for
###100 abstracts. This was done so that 10 sets of 100 abstracts could have their shap values calculated in parallel (10 different machines).
###mistral31_word_shap2 uses strongly predicted retracted papers (>75%) to create a dictionary of top words.
###mistral31_word_shap uses a simple regex tokenizer to use whole words rather than tokens from Mistral.
###mistral31_shap attempts to explain 1-10 abtracts as to why the model predicted they were retracted. This requires
###the shap_env environment as that is where shap is installed.
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import shap
from transformers import AutoTokenizer, AutoModel
import torch
import csv
import pickle
import re
import joblib
import random
import argparse

model_name = "mistralai/Mistral-7B-v0.1"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

class WordTokenizer:
    def __init__(self):
        self.pattern = r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?"
        self.special_tokens_map = {}
    def __call__(self, text, return_offsets_mapping=False, **kwargs):
        if text == "":
            out = {"input_ids": [], "offset_mapping": []}
            return out
        matches = list(re.finditer(self.pattern, text))
        if len(matches) == 0:
            out = {"input_ids": ["MASK"],"offset_mapping": [(0, len(text))]}
            return out
        tokens = [m.group(0) for m in matches]
        offsets = [(m.start(), m.end()) for m in matches]
        out = {"input_ids": tokens}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        else:
            out["offset_mapping"] = offsets
        return out
    def decode(self, ids):
        if isinstance(ids, list):
            return " ".join(str(x) for x in ids)
        return str(ids)

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    ###forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states
    ###final layer hidden states: shape (1, seq_len, hidden_dim)
    last_layer = hidden_states[-1]
    
    ###embedding = last_layer.mean(dim=1)
    ###return embedding.cpu().numpy()
    last_embedding = last_layer[0, -1, :]
    return last_embedding.cpu().numpy()

#this class holds the information for each abstract
class Node:
    def __init__(self):
        self.embedding = []
        self.num_words = 0
        self.retracted_or_not_retracted = -1   ###retracted = 1, not-retracted = 0
        self.date = ''
    def __repr__(self):
        return f'Node({self.error_or_fraud}, {self.num_words}, {self.embedding})'
    
def predict_retraction(texts):
    embeddings = []
    for abstract in texts:
        emb = get_embeddings(abstract)
        embeddings.append(emb)
    embeddings = scaler.transform(np.array(embeddings))
    return classifier.predict_proba(embeddings)[:,1]

if __name__ == "__main__":
    print(datetime.datetime.now())
    parser = argparse.ArgumentParser(description='SHAP input range')
    parser.add_argument("-i", "--shap_100_index", default=None, type=str, help="load jit model")
    args = parser.parse_args()
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()
    early_papers = []
    
    with open('retraction_watch_embeddings30_mistral_early.csv', 'r') as file:
    ###with open('retraction_watch_embeddings30_mistral_early_no_citations.csv', 'r') as file:
    ###with open('retraction_watch_embeddings30_lora_batch8_epochs5_early.csv', 'r') as file:
    ###with open('retraction_watch_embeddings30_ia3_8batch_5epochs_early.csv', 'r') as file:
        for i, line in enumerate(file):
            ###replace elements of the abstract with nothing so that it doesn't confuse ChatGPT
            line = line.replace('[','').replace(']','').replace("\n",'')
            if line != '':
                paper = Node()
                line_split = []
                line = line.split(',')
                for element in line:
                    line_split.append(float(element.strip().replace('"','').replace("'","")))
                paper.embedding = line_split
                paper.retracted_or_not_retracted = 1
                early_papers.append(paper)
    with open('semantic_scholar_embeddings30_mistral_early.csv', 'r') as file:
    ###with open('semantic_scholar_embeddings30_mistral_early_no_citations.csv', 'r') as file:
    ###with open('semantic_scholar_embeddings30_lora_batch8_epochs5_early.csv', 'r') as file:
    ###with open('semantic_scholar_embeddings30_ia3_8batch_5epochs_early.csv', 'r') as file:

        for i, line in enumerate(file):
            ###replace elements of the abstract with nothing so that it doesn't confuse ChatGPT
            line = line.replace('[','').replace(']','').replace("\n",'')
            if line != '':
                paper = Node()
                line_split = []
                line = line.split(',')
                for element in line:
                    line_split.append(float(element.strip().replace('"','').replace("'","")))
                paper.embedding = line_split
                paper.retracted_or_not_retracted = 0
                early_papers.append(paper)
    recent_papers = []
    
    with open('retraction_watch_embeddings30_mistral_recent.csv', 'r') as file:
    ###with open('retraction_watch_embeddings30_mistral_recent_no_citations.csv', 'r') as file:
    ###with open('retraction_watch_embeddings30_lora_batch8_epochs5_recent.csv', 'r') as file:
    ###with open('retraction_watch_embeddings30_ia3_8batch_5epochs_recent.csv', 'r') as file:
        for i, line in enumerate(file):
            ###replace elements of the abstract with nothing so that it doesn't confuse ChatGPT
            line = line.replace('[','').replace(']','').replace("\n",'')
            if line != '':
                paper = Node()
                line_split = []
                line = line.split(',')
                for element in line:
                    line_split.append(float(element.strip().replace('"','').replace("'","")))
                paper.embedding = line_split
                paper.retracted_or_not_retracted = 1
                recent_papers.append(paper)
    with open('semantic_scholar_embeddings30_mistral_recent.csv', 'r') as file:
    ###with open('semantic_scholar_embeddings30_mistral_recent_no_citations.csv', 'r') as file:
    ###with open('semantic_scholar_embeddings30_lora_batch8_epochs5_recent.csv', 'r') as file:
    ###with open('semantic_scholar_embeddings30_ia3_8batch_5epochs_recent.csv', 'r') as file:
        for i, line in enumerate(file):
            ###replace elements of the abstract with nothing so that it doesn't confuse ChatGPT
            line = line.replace('[','').replace(']','').replace("\n",'')
            if line != '':
                paper = Node()
                line_split = []
                line = line.split(',')
                for element in line:
                    line_split.append(float(element.strip().replace('"','').replace("'","")))
                paper.embedding = line_split
                paper.retracted_or_not_retracted = 0
                recent_papers.append(paper)

    #use the early papers, before text-embedding-3-small was trained, for training and the recent papers as test
    scaler = StandardScaler()
    early_labels = np.array([early_papers[i].retracted_or_not_retracted for i in range(len(early_papers))])
    print("early_labels",early_labels.shape)
    len_input = len(early_labels)
    np.random.seed(17)
    ###10% of input used for validation, but requires /2 additional as both retracted and valid papers are included,
    ###and they have to be put into validation together in pairs to not introduce bias
    indices_first_half = np.sort(np.random.choice(len_input//2, size=len_input//20, replace=False))
    ###print(indices_first_half)
    indices = np.concatenate((indices_first_half,np.array([i+len_input//2 for i in indices_first_half])))
    Xtrain = scaler.fit_transform(np.array([early_papers[i].embedding for i in range(len(early_papers)) if i not in indices]))
    Ytrain = np.array([early_papers[i].retracted_or_not_retracted for i in range(len(early_papers)) if i not in indices])
    print("len Xtrain Ytrain",len(Xtrain),len(Ytrain))
    Xval = scaler.transform(np.array([early_papers[i].embedding for i in indices]))
    Yval = np.array([early_papers[i].retracted_or_not_retracted for i in indices])
    print("len Xval Yval",len(Xval),len(Yval))
    Xtest_full = scaler.transform(np.array([recent_papers[i].embedding for i in range(len(recent_papers))]))
    Ytest_full = np.array([recent_papers[i].retracted_or_not_retracted for i in range(len(recent_papers))])
    print("len Xtest Ytest",len(Xtest_full),len(Ytest_full))
    
    ###print("mistral: train with valid high citation papers, test with random papers ")
    ###QLoRA
    ###for C in [1., 10., 100.]:
    ###    for gamma in [1e-3, 1e-4, 1e-5]:
    C = 100. 
    gamma = 0.0001
    classifier = svm.SVC(kernel="rbf", class_weight = 'balanced', C=C, gamma=gamma, probability=True)
    classifier.fit(Xtrain, Ytrain)
    ###joblib.dump(classifier, "svm_classifier.joblib")
    ###joblib.dump(scaler, "scaler.joblib")
    y_pred = classifier.predict_proba(Xtest_full)[:,1]
    all_indices = []
    with open('retracted_1000_indices.csv', 'r') as f:
        for i, line in enumerate(f):
            line = line.strip('"').strip('[').strip(']').strip(' ').strip('\n').split()
            line = [int(x[:-2]) if x.endswith(']"') else int(x) for x in line]
            all_indices += line
    sampled_indices = all_indices[int(args.shap_100_index)*100:(int(args.shap_100_index)+1)*100]
    print("sampled indices", sampled_indices)
    
    abstracts = []
    with open('retraction_watch_abstract30_recent.csv', 'r', encoding="utf8") as file:
    ###with open('retraction_watch_abstract30_recent_no_citations.csv', 'r', encoding="utf8") as file, open('retraction_watch_embeddings30_mistral_recent_no_citations.csv', 'w') as f:
        for i, line in enumerate(file):
            if i % 1000 == 0:
                print(i,datetime.datetime.now())
            if len(line) > 1:
                abstract = line[line.find('Abstract')+9:]
                abstract = abstract.replace('\n','').replace('"','')
                abstracts.append(abstract)
    
    ###masker = shap.maskers.Text(tokenizer)
    word_tokenizer = WordTokenizer()
    masker = shap.maskers.Text(word_tokenizer)
    explainer = shap.Explainer(predict_retraction, masker)

    words_shap_values = {}
    for i, idx in enumerate(sampled_indices):
        if i % 10 == 0:
            print("i idx", i, idx)
        shap_values = explainer([abstracts[idx]])
        values = np.array(shap_values[0].values).reshape(-1)
        ###print("values",values)
        words = shap_values[0].data
        ###print("words", words)
        top_value_indices = np.argsort(np.abs(values))[-1000:]
        for j, word in enumerate(words):
            word = str(word).strip().lower()   ###strip spaces and convert to lower case
            word = re.sub(r"^[^\w]+|[^\w]+$", "", word)   ###strip punctuation at start/end
            word = re.sub(r"\s+", " ", word)   ###collapse internal spaces
            if word in words_shap_values and word != "":
                words_shap_values[word].append(values[j])
            else:
                words_shap_values[word] = [values[j]]
    ###print("words_shap_values", words_shap_values)
    ###words_ave_shap_value = {}
    with open('retracted_words_shap_values_100_'+args.shap_100_index+'.csv', 'w') as f:
        writer = csv.writer(f)
        for j, (key, value) in enumerate(words_shap_values.items()):
            if len(value) > 1:
                print("key len(value) value sum(value)", key, len(value), value, sum(value))
            ###writer.writerow([key, sum(value)/len(value)])
            writer.writerow([key, len(value), value, sum(value)])
            ###words_ave_shap_value[key] = sum(value)/len(value)
    '''
    top_shap_words = sorted(words_ave_shap_value, key=lambda k: abs(words_ave_shap_value[k]), reverse=True)[:20]
    for word in top_shap_words:
        print(word, words_ave_shap_value[word])
    '''
    print(datetime.datetime.now())
    
