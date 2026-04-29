###few_shot.py provides one random valid paper abstract and one random retracted paper abstract, and asks OpenAI 3.5 to return the probability 
###that a third paper abstract is from a retracted paper.
import csv
import datetime
from openai import OpenAI
import time
import random

client = OpenAI(api_key="")
primes = [17,41,97,113,179]
###This detailed prompt gives background, describes the task of identifying what a retracted paper is and why we're asking, an example of a valid paper and an example
###of a valid paper, and asks for a probability to be returned on whether the supplied abstract is from a retracted paper.
def abstract_retraction_probability(valid_abstract, retracted_abstract, test_abstract, model="gpt-4.1"):
    prompt = f"""
    ## Task Background:
    The scientific community has always had instances of published works that had errors in the data, methods, analysis, or conclusions of the authors work. 
    If this is discovered after a paper is published, the paper is retracted  and marked as such. When papers are retracted, it adversely affects the progress 
    of new research, and also affects the trust of the general public in the scientific community. In some cases the paper is retracted because of an error in
    the authors reasoning, analysis, or conclusions, but sometimes it is because of malfeasance on the part of the author. Taking a step towards detecting 
    fraudulent or erroneous papers prior to publication can minimize this impact. 
    ## Task Instructions:
    You will receive an example abstract for a valid research paper and an example abstract for a retracted research paper. Your task is to determine the 
    **probability** that the abstract for a third research paper **will be retracted** or **has already been retracted**.

    - Respond with **only a number** between **0 and 1**, inclusive.
    - **Do not include any explanation, units, or commentary.**
    - If uncertain, make your best estimate based on the text.
    - The response must be in this format:
    0.42
    ## Examples:
    example abstract for a valid research paper {valid_abstract}, Label: probability close to 0.0
    example abstract for a retracted research paper {retracted_abstract}, Label: probability close to 1.0
    ## Test Abstract:
    Please classify this abstract {test_abstract}
    """
    
    response = client.chat.completions.create(model=model,messages=[{"role":"system","content":"""You are an expert NLP assistant. You are not permitted to use internet access,
                                                              training data, or prior world knowledge. Your response must be based solely on the text I provide below."""},
                                                              {"role":"user","content":prompt}])
    retraction_probability = response.choices[0].message.content.replace('\n', ' ')
    return retraction_probability

if __name__ == "__main__":
    print(datetime.datetime.now())
    
    valid_abstracts = []
    retracted_abstracts = []
    with open('semantic_scholar_abstract30_recent.csv', 'r') as f, open('semantic_scholar_few_shot30.csv', 'w') as f2, \
        open('retraction_watch_abstract30_recent.csv', 'r') as f3, open('retraction_watch_few_shot30.csv', 'w') as f4:
        writer2 = csv.writer(f2)
        writer4 = csv.writer(f4)
        print("getting valid abstracts")
        for i, line in enumerate(f):
            if len(line) > 0:
                abstract = line.strip().replace('\n',' ')
                valid_abstracts.append(abstract)
            else:
                print("len semantic_scholar line = 0", i)
        print("getting retracted abstracts")
        for j, line2 in enumerate(f3):
            if len(line2) > 0:
                abstract = line2[line2.find('Abstract')+9:].strip().replace('\n',' ')
                retracted_abstracts.append(abstract)
            else:
                print("len retraction_watch line = 0", j)
        print("getting probabilities")
        for k in range(len(valid_abstracts)):
            i_example = random.choice([m for m in range(len(valid_abstracts)) if m != k])
            j_example = random.choice([n for n in range(len(valid_abstracts)) if n != k])
            retraction_probability = abstract_retraction_probability(valid_abstracts[i_example],retracted_abstracts[j_example],valid_abstracts[k])
            writer2.writerow([retraction_probability])
            time.sleep(1)
            retraction_probability = abstract_retraction_probability(valid_abstracts[i_example],retracted_abstracts[j_example],retracted_abstracts[k])
            writer4.writerow([retraction_probability])
            time.sleep(1)
            if k % 100 == 0:
                print(k,datetime.datetime.now(),"retraction_probability",retraction_probability)

    print("\n",datetime.datetime.now())