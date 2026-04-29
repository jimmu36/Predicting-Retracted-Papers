###zero_shot.py simply asks OpenAI 3.5 to return the probability an abstract is from a retracted paper.
import numpy as np
import pandas as pd
import csv
import re
import requests
from bs4 import BeautifulSoup
import os
import datetime
from openai import OpenAI
import time

client = OpenAI(api_key="")
primes = [17,41,97,113,179]
###This detailed prompt gives background, describes the task of identifying what a retracted paper is and why we're asking, and asks for a probability to be
###returned on whether the supplied abstract is from a retracted paper.
def abstract_retraction_probability(seed, abstract, model="gpt-4.1"):
    prompt = f"""
    ## Task Background:
    The scientific community has always had instances of published works that had errors in the data, methods, analysis, or conclusions of the authors work. 
    If this is discovered after a paper is published, the paper is retracted  and marked as such. When papers are retracted, it adversely affects the progress 
    of new research, and also affects the trust of the general public in the scientific community. In some cases the paper is retracted because of an error in
    the authors reasoning, analysis, or conclusions, but sometimes it is because of malfeasance on the part of the author. Taking a step towards detecting 
    fraudulent or erroneous papers prior to publication can minimize this impact. 
    ## Task Instructions:
    You will receive the abstract of a research paper. Your task is to determine the **probability** that the paper **will be retracted** or **has already been retracted**.

    - Respond with **only a number** between **0 and 1**, inclusive.
    - **Do not include any explanation, units, or commentary.**
    - If uncertain, make your best estimate based on the text.
    - The response must be in this format:
    0.42
    ## Text:
    \"\"\"{abstract}\"\"\"
    """
    
    response = client.chat.completions.create(model=model,seed=seed,messages=[{"role":"system","content":"""You are an expert NLP assistant. You are not permitted to use internet acess,
                                                              training data, or prior world knowledge. Your response must be based solely on the text I provide below."""},
                                                              {"role":"user","content":prompt}])
    retraction_probability = response.choices[0].message.content.replace('\n', ' ')
    return retraction_probability

if __name__ == "__main__":
    print(datetime.datetime.now())
    
    print("semantic scholar")
    with open('semantic_scholar_abstract30_recent.csv', 'r') as f, open('semantic_scholar_zero_shot30.csv', 'w') as f2:
    ###with open('semantic_scholar_abstract30_mistral_recent.csv', 'r') as f, open('semantic_scholar_zero_shot30.csv', 'w') as f2:
        writer = csv.writer(f2)
        for i, line in enumerate(f):
            ave = 0
            ###for seed in [17]:
            for seed in primes:
                if len(line) > 0:
                    abstract = line.strip().replace('\n',' ')
                    if float(abstract_retraction_probability(seed, abstract)) >= 0.5:
                        ave += 1
                else:
                    print("len semantic_scholar line = 0", i)
                time.sleep(1)
            ###writer.writerow([ave])
            writer.writerow([ave/5])
            if i % 100 == 0:
                print(i,datetime.datetime.now(),"retraction_probability",ave)
            
    print("retraction watch")
    with open('retraction_watch_abstract30_recent.csv', 'r') as f, open('retraction_watch_zero_shot30.csv', 'w') as f2:
    ###with open('retraction_watch_abstract30_mistral_recent.csv', 'r') as f, open('retraction_watch_zero_shot30.csv', 'w') as f2:
        writer = csv.writer(f2)
        for i, line in enumerate(f):
            ave = 0
            ###for seed in [17]:
            for seed in primes:
                if len(line) > 0:
                    abstract = line[line.find('Abstract')+9:].strip().replace('\n',' ')
                    if float(abstract_retraction_probability(seed, abstract)) >= 0.5:
                        ave += 1
                else:
                    print("len retraction_watch line = 0", i)
                time.sleep(1)
            ###writer.writerow([ave])
            writer.writerow([ave/5])
            if i % 100 == 0:
                print(i,datetime.datetime.now(),"average retraction_probability",ave)
                
    print("\n",datetime.datetime.now())