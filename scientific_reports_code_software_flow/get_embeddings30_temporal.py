###_no_citations uses abstracts without sorting the citations to get the highest cited papers. retracted found 18605, similar valid
###papers found 12128, recent 6114, early 6014.
###get_embeddings30_temporal follows from get_embeddings16_temporal_retracted_not_retracted_test2. It uses get_abstract30_temporal,
###which deletes papers with "retract" or with "withdrawn" in the abstract.
###The _temporal_retracted_not_retracted version makes sure the similar paper is in the early papers (before 2021 or earlier as OpenAI was
###generally made available by end of Sept '21) or later papers (2022 or later) year ranges.
###get_abstract16 uses from year 2001 - 2/15/25 and all subjects rather than the limitations in get_abstract14/15.
###get_abstract15 uses semantic scholar to get similar abstracts rather than openai in get_abstracts14. The goal will be to predict whether
###a paper is a retracted paper or not, rather than predicting if a retracted paper was due to error or fraud.
###get_abstract14 follows from get_abstract10. It tries to duplicate the dataset of the paper "Understanding and Predicting Retractions of 
###Published Work", which only uses subjects Health Sciences (HSC, 5,396 papers), Social Sciences (SOC, 2,651 papers), and Humanities 
###(HUM, 366 papers) from 2001 to 2019.
###get_embeddings10 records the reasons for the retraction as well, which will allow the use of the embeddings for multiple
###tests. All fraud/error reasons, focus (top 13fraud/6error reasons), focus 2 (top 6fraud/6error reasons), data, and result reasons.
###get_retractions_abstract_embeddings3 adds the publication date so that frauds/errors can be checked
###before the date/after the date of the LLM embedding (to make sure the LLM had not seen that paper
###as part of the training of the LLM).
from openai import OpenAI
import datetime
import numpy as np
import csv
import time

def get_chatgpt_embedding(text):
    client = OpenAI(api_key="")
    return client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding

###Version 3 includes the original paper date so that test cases can be from after the OpenAI embeddings model text-embedding-3-small
###was created (Sept '21).
if __name__ == "__main__":
    print(datetime.datetime.now())
    
    ###with open('retraction_watch_abstract30_recent.csv', 'r', encoding="utf8") as file, open('retraction_watch_embeddings30_recent.csv', 'w') as f:
    with open('retraction_watch_abstract30_recent_no_citations.csv', 'r', encoding="utf8") as file, open('retraction_watch_embeddings30_recent_no_citations.csv', 'w') as f:
        writer = csv.writer(f)
        for i, line in enumerate(file):
            if i % 1000 == 0:
                print(i,datetime.datetime.now())
            ###print(i,end=".")
            if len(line) > 1:
                abstract = line[line.find('Abstract')+9:]
                abstract = abstract.replace('\n','').replace('"','')
                embedding = get_chatgpt_embedding(abstract)
                writer.writerow([embedding])
            time.sleep(1)
    
    ###with open('semantic_scholar_abstract30_recent.csv', 'r', encoding="utf8") as file, open('semantic_scholar_embeddings30_recent.csv', 'w') as f:
    with open('semantic_scholar_abstract30_recent_no_citations.csv', 'r', encoding="utf8") as file, open('semantic_scholar_embeddings30_recent_no_citations.csv', 'w') as f:
        writer = csv.writer(f)
        for i, line in enumerate(file):
            if i % 1000 == 0:
                print(i,datetime.datetime.now())
            ###print(i,end=".")
            if len(line) > 0:
                abstract = line.replace('\n','').replace('"','')
                embedding = get_chatgpt_embedding(abstract)
                writer.writerow([embedding])
            time.sleep(1)
    
    ###with open('retraction_watch_abstract30_early.csv', 'r', encoding="utf8") as file, open('retraction_watch_embeddings30_early.csv', 'w') as f:
    with open('retraction_watch_abstract30_early_no_citations.csv', 'r', encoding="utf8") as file, open('retraction_watch_embeddings30_early_no_citations.csv', 'w') as f:
        writer = csv.writer(f)
        for i, line in enumerate(file):
            if i % 1000 == 0:
                print(i,datetime.datetime.now())
            ###print(i,end=".")
            if len(line) > 1:
                abstract = line[line.find('Abstract')+9:]
                abstract = abstract.replace('\n','').replace('"','')
                embedding = get_chatgpt_embedding(abstract)
                writer.writerow([embedding])
            time.sleep(1)
    
    ###with open('semantic_scholar_abstract30_early.csv', 'r', encoding="utf8") as file, open('semantic_scholar_embeddings30_early.csv', 'w') as f:
    with open('semantic_scholar_abstract30_early_no_citations.csv', 'r', encoding="utf8") as file, open('semantic_scholar_embeddings30_early_no_citations.csv', 'w') as f:
        writer = csv.writer(f)
        for i, line in enumerate(file):
            if i % 1000 == 0:
                print(i,datetime.datetime.now())
            ###print(i,end=".")
            if len(line) > 0:
                abstract = line.replace('\n','').replace('"','')
                embedding = get_chatgpt_embedding(abstract)
                writer.writerow([embedding])
            time.sleep(1)
    
    print("\n",datetime.datetime.now())