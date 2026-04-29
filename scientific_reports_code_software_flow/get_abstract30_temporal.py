###get_abstract30_temporal follows from get_abstract16_temporal_retracted_not_retracted_test2. It deletes papers with "retract" (or any form 
###of the word) or with "withdrawn" in the abstract.
###The _temporal_retracted_not_retracted version makes sure the similar paper is in the early papers (before 2021 or earlier as OpenAI was
###generally made available by end of Sept '21) or later papers (2022 or later) year ranges.
###get_abstract16 uses from year 2001 - 2/15/25 and all subjects rather than the limitations in get_abstract14/15.
###get_abstract15 uses semantic scholar to get similar abstracts rather than openai in get_abstracts14. The goal will be to predict whether
###a paper is a retracted paper or not, rather than predicting if a retracted paper was due to error or fraud.
###get_abstract14 follows from get_abstract10. It tries to duplicate the dataset of the paper "Understanding and Predicting Retractions of 
###Published Work", which only uses subjects Health Sciences (HSC, 5,396 papers), Social Sciences (SOC, 2,651 papers), and Humanities 
###(HUM, 366 papers) from 2001 to 2019.
###get_abstract10 records the reasons for the retraction as well. This will enable get_embeddings10 to get all of the embeddings and
###save those along with the retraction reasons in openai_embeddings10, which will allow the use of the embeddings for multiple
###tests. All fraud/error reasons, focus (top 13fraud/6error reasons), focus 2 (top 6fraud/6error reasons), data, and result reasons.
###Version 4 uses the pub med id site. The problem is there are far fewer OriginalPaperPubMedID papers.
###Version 3 works with the DOI site to get the abstract (the earlier version were using the record ID from the retraction watch database,
###which was not correct). The problem is the DOI site asks for human verification, so that's not going to work more than one request with
### a long delay.
###Version 2 includes the original paper date so that test cases can be from after the OpenAI embeddings model text-embedding-3-small
###was created (Sept '21).
import numpy as np
import pandas as pd
import csv
import re
import requests
from bs4 import BeautifulSoup
import os
import datetime
###from openai import OpenAI
import time

###This routine only needs to be run once for each new version of the Retraction Watch database. It reads in each line of the
###retraction database in pandas format that is passed in, and parses the Reason field. There usually is many more than one
###reason for the rectraction. It stores each retraction in the reasons dictionary, and keeps count of how many times each
###reason appears in the database.
def count_reasons(data):
    reasons = {}
    for i in range(len(data)):
        ###If there is nothing in the field, it returns NaN, but you can't check on NaN, you have to check if it is a float. Otherwise, the return is
        ###a str.
        if isinstance(data['Reason'][i], float):
            print('*')
        else:
            reasons_this_retraction = data['Reason'][i].split('+')
            for reason in reasons_this_retraction:
                if reason != '':
                    if reason in reasons:
                        reasons[reason] += 1
                    else:
                        reasons[reason] = 1
    return reasons

def get_recent(date_this_line):
    ###print("date_this_line", date_this_line, isinstance(date_this_line, int))
    dashes = [i for i, letter in enumerate(date_this_line) if letter == '/']
    recent = '9/30/2021' 
    test_year = int(date_this_line[dashes[1]+1:dashes[1]+5])
    test_month = int(date_this_line[0:dashes[0]])
    ###print("recent[5:]", int(recent[5:]), "test_year", test_year, test_year > int(recent[5:]))
    ###print("month", int(recent[0]), "test_month", test_month, test_month > int(recent[0]))
    if (test_year > int(recent[5:])) or ((test_year == int(recent[5:])) and (test_month > int(recent[0]))):
    ###recent = '12/31/2023' 
    ###if (date_this_line[dashes[1]+1:dashes[1]+5] > recent[6:]) or \
    ###    (date_this_line[dashes[1]+1:dashes[1]+5] == recent[6:] and date_this_line[0:dashes[0]] > recent[0]):
            return True
    else:
        ###print("date_this_line", date_this_line, "test_year", test_year, "test_month", test_month)
        return False

###This routine relies on the user must manually editing retraction_watch_2-15_count.csv to set the specific reasons as 0=error, 1=fraud, 2=indeterminate
###in column 3. This will vary depending on which reasons the user designates to each category. The goal is to get a dictionary of reasons that are either
###error or fraud and can be searched. These dictionaries could have been created by typing them in from the known reasons in the Retraction Watch database,
###but this eliminates all the typing and potential errors.
def fraud_error_reasons(csv_file):
    fraud_reasons = {}
    error_reasons = {}
    with open(csv_file,'r') as f:
        for i, line in enumerate(f):
            ###the semicolun is in the Retract Watch database and needs to be removed to make the comparisons work
            line = line.strip().replace(';','').split(',')
            ###print(line[2])
            if line[2] == '1':
                fraud_reasons[line[0]] = 1
            elif line[2] == '0':
                error_reasons[line[0]] = 1
    return fraud_reasons, error_reasons

###This routine returns the report ID which is used to find the abstract of the paper in the Retraction Watch database. Since it is looping
###through the whole database, it also keeps track of which papers were retracted for which reason 0=error, 1=fraud, 2=indeterminate.
###However, if the retraction reason was 2=indeterminate, then the report ID is dropped and not processed further/not returned.
def get_OriginalPaperPubMedID_reasons(data, fraud_reasons, error_reasons):
    fraud_error_count = {0: 0, 1: 0, 2: 0}  ###0=error, 1=fraud, 2=indeterminate
    OriginalPaperPubMedID_reason_dates = []
    for i in range(len(data)):
        ###If there is nothing in the field, it returns NaN, but you can't check on NaN, you have to check if it is a float. Otherwise, the return is
        ###a str.
        if isinstance(data['Reason'][i], float):
            print('*')
        elif data['OriginalPaperPubMedID'][i] != 0:
            ###The semicolun is in the Retract Watch database and needs to be removed to make the comparisons work. The reasons field usually has
            ###more than one reason for the retraction, and they are separated by '+'.
            reasons_this_retraction = data['Reason'][i].replace(';','').split('+')
            fraud_error = 2  ###default is indeterminate
            for reason in reasons_this_retraction:
                if reason in fraud_reasons:  ###if any reason in the reason field is fraud, the paper is deemed to be retracted for fraud
                    fraud_error = 1
                ###If not yet declared retracted for fraud, then the reason is declared error if it is in the error_reasons dictionary. However, if a later
                ###reason in this same line of the Retraction Watch database is in the fraud_reasons dictionary, the fraud designation will overwrite this
                ###error designation.
                if fraud_error == 2 and reason in error_reasons:  
                    fraud_error = 0
            fraud_error_count[fraud_error] += 1  ###fraud_error_count dictionary indexed by 0=error, 1=fraud, 2=indeterminate
            ###If there is nothing in the field, it returns NaN, but you can't check on NaN, you have to check if it is a float. Otherwise, the return is
            ###a str.
            ###if i < 10:
            ###    print(data['Record ID'][i],data['OriginalPaperPubMedID'][i], data['OriginalPaperPubMedID'][i] != 0,"fraud_error",fraud_error, fraud_error != 2)
            ###if not isinstance(data['OriginalPaperPubMedID'][i], float) and (data['OriginalPaperPubMedID'][i] != 0) and (fraud_error != 2):
            if fraud_error != 2:
                OriginalPaperPubMedID_reason_dates.append([int(data['OriginalPaperPubMedID'][i]), fraud_error, data['OriginalPaperDate'][i], data['Reason'][i]])
    return OriginalPaperPubMedID_reason_dates, fraud_error_count

def get_similar_nonRetracted_abstract(abstract,id,similar_abstracts, date_this_line):
    ###Define the API endpoint URL for getting the Semantic Scholar paperId
    url = f"http://api.semanticscholar.org/graph/v1/paper/{id}"
    ###print("url in get_similar", url)
    # Directly define the API key for Semantic Scholar
    api_key = "tAPinDnNtP1ydCM0Dm4137he3m1RQzYF1qOzXNFW"  # Replace with the actual API key
    # Define headers with API key
    headers = {"x-api-key": api_key}
    # Send the API request
    ###response = requests.get(url, params=query_params, headers=headers)
    response = requests.get(url, headers=headers)
    ###print("response",response)
    time.sleep(2)   ###need to add sleep as only 100 inquiries per 5min/1 inquiry per sec is allowed
    if response.status_code == 200:   ###Check response status, 200 is successful
        response_data = response.json()
        ###print("response_data['paperId']",response_data['paperId'])
        ###Define the API endpoint URL for getting 10 similar papers from Semantic Scholar.
        url = "https://api.semanticscholar.org/recommendations/v1/papers/forpaper/"+response_data['paperId']
        ###Define the query parameters for the top 10 papers, which will be sorted from highest to lowest citation count,
        ###and then the first one with an abstract will used as the similar paper.
        recent_retracted_paper = get_recent(date_this_line)
        ###print("date_this_line", date_this_line, "recent?", recent_retracted_paper)
        if recent_retracted_paper:
            query_params = {"fields": "citationCount,abstract,url,year,externalIds",
                            "limit": "20",
                            "pool-from": 'recent'}
        else:
            query_params = {"fields": "citationCount,abstract,url,year,externalIds",
                            "limit": "20",
                            "year": '2001-2021'}
        response2 = requests.get(url, params=query_params, headers=headers)
        ###print("response2",response2)
        if response2.status_code == 200:   ###Check response status, 200 is successful
            response2 = response2.json()
            papers = response2["recommendedPapers"]
            ###print("papers", len(papers))
            if len(papers) > 0:
                ###Sort the 10 abstracts highest citation count to lowest citation count.
                papers.sort(key=lambda paper: paper["citationCount"], reverse=True)
                ###Use the top rated citation paper that meets the criterion: an abstract exits; length > 100 is a proxy for
                ###the similar paper being just another paper that says why the retracted paper is in error; the paper cannot
                ###already have been selected as a similar paper for a previous retracted paper.
                for i, paper in enumerate(papers):
                    if 'DOI' in paper["externalIds"]:
                        doi = paper["externalIds"]["DOI"]
                        retracted = False
                        response3 = requests.get(f"https://api.crossref.org/v1/works/{doi}", timeout=30)
                        if response3:
                            response3 = response3.json()
                            if 'message' in response3:
                                messages = response3["message"]
                                for message in messages.get("update-to", []):
                                    if message.get("type") == "retraction" or message.get("source") == "retraction-watch":
                                        retracted = True
                                if paper["year"] != None and paper["abstract"] != None:
                                    abstract_stripped = paper['abstract'].replace('\n',' ').replace('"','').strip()
                                    if paper['abstract'].find('Not available') == -1 and len(paper["abstract"]) > 100 and abstract_stripped not in similar_abstracts \
                                        and 'retract' not in paper["abstract"].lower() and 'withdrawn' not in paper["abstract"].lower() and not retracted:
                                        return abstract_stripped, recent_retracted_paper
    return None, None   ###if the critierion are not met or the response status was not successful

if __name__ == "__main__":
    print(datetime.datetime.now())
    ###Get the fraud and error reasons in the actual database. This is only used to drop the indeterminate reason abstracts.
    fraud_reasons, error_reasons = fraud_error_reasons('retraction_watch_2-15_count_no_blank_pubmedid16.csv')
    print("len(fraud_reasons) len(error_reasons)",len(fraud_reasons), len(error_reasons))
    data = pd.read_csv('retraction_watch_2-15_no_blank_pubmedid16.csv')
    ###In order to search for paper abstracts, the report ID number needs to be extracted from the Retractions Watch database. Since the routine retrieving the
    ###report ID is parsing each line, it can also keep track of the number of each instance of 0=error, 1=fraud, 2=indeterminate.
    OriginalPaperPubMedID_reason_dates, fraud_error_count = get_OriginalPaperPubMedID_reasons(data, fraud_reasons, error_reasons)
    print("errors, frauds, indeterminates", fraud_error_count[0], fraud_error_count[1], fraud_error_count[2])
    print("len(OriginalPaperPubMedID_reason_dates)", len(OriginalPaperPubMedID_reason_dates))
    
    retracted_abstracts_found = 0
    similar_abstracts_found = 0
    recent_abstracts_found = 0
    early_abstracts_found = 0
    similar_abstracts = []
    ###The Beautiful Soup code here was modified from what Chaturya, an Asa Ben-Hur student, previously wrote. If there's no abstract associated
    ###with the page for the report ID, or if the abstract field says "No abstract available", the report ID and associated reason for retraction
    ###are dropped (not processed further). Only if an abstract is found, then the reason for retraction 0=error or 1=fraud, the length of the 
    ###abstract in words, and the abstract itself are written to a file. Note that the retraction reason 2=indeterminate was already dropped by
    ###the get_OriginalPaperPubMedID_reasons function. The number of abstracts found is kept so that it can be compared with the overall number of report IDs 
    ###having retraction reason of error=0 or fraud=1.
    for i, OriginalPaperPubMedID_reason_date in enumerate(OriginalPaperPubMedID_reason_dates):
        if i % 100 == 0:
            print(i, "retracted_abstracts_found, similar_abstracts_found, recent_abstracts_found, early_abstracts_found",
                  retracted_abstracts_found, similar_abstracts_found,recent_abstracts_found, early_abstracts_found, datetime.datetime.now())
        # Construct the URL for the abstract page on the DOI website
        url = f"https://pubmed.ncbi.nlm.nih.gov/{OriginalPaperPubMedID_reason_date[0]}/"
        ###print(url)
        response = requests.get(url, timeout=(30, 30))
        soup = BeautifulSoup(response.content, "html.parser")
        doi_link = soup.find("a", {"class": "id-link"})
        doi = doi_link.text if doi_link else None
        # Construct the URL for the abstract page on the DOI website
        if doi:
            doi = doi.strip()
            doi_url = f"https://doi.org/{doi}"
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        ###abstract = soup.find('div', {'class': 'article-section__content en main'})
        abstract = soup.find('div', {'class': 'abstract'}) or soup.find('div' , {'class': 'Abstact'})
        if abstract and abstract.text.strip()[:21] != "No abstract available" and 'retract' not in abstract.text.strip().lower() and 'withdrawn' not in abstract.text.strip().lower():
            retracted_abstracts_found += 1
            abstract_text = re.sub(r"\s+", " ", abstract.text.strip())
            abstract_text = abstract_text.replace('[','').replace(']','').replace("'","").replace('\n','').replace('"\n','').strip()
            ###If there's a retracted abstract, get a similar abstract from Semantic Scholar.
            id = f'PMID:{OriginalPaperPubMedID_reason_date[0]}'
            ###The _temporal_retracted_not_retracted version passes the date in as well so that the papers can be split
            ###into recent (after Sept '21 when OpenAI became generally available, value True) and early (value False).
            similar_abstract, recent_paper = get_similar_nonRetracted_abstract(abstract_text,id,similar_abstracts,OriginalPaperPubMedID_reason_date[2])
            if similar_abstract != None:
                similar_abstracts_found += 1
                if recent_paper:
                    
                    with open('retraction_watch_abstract30_recent.csv','a',encoding='utf-8') as f:
                        writer = csv.writer(f)
                        ###[0] = PubMed ID, [1] = 1/0 fraud/error, [2] = OrigPaperDate, [3] = list of reasons for retraction
                        ###writer.writerow([OriginalPaperPubMedID_reason_date[1],OriginalPaperPubMedID_reason_date[3],abstract_text,OriginalPaperPubMedID_reason_date[2]])
                        writer.writerow([OriginalPaperPubMedID_reason_date[1],OriginalPaperPubMedID_reason_date[3],abstract_text])
                    
                    recent_abstracts_found += 1
                    similar_abstracts.append(similar_abstract)
                    
                    with open('semantic_scholar_abstract30_recent.csv','a',encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([similar_abstract])
                    
                else:
                    
                    with open('retraction_watch_abstract30_early.csv','a',encoding='utf-8') as f:
                        writer = csv.writer(f)
                        ###[0] = PubMed ID, [1] = 1/0 fraud/error, [2] = OrigPaperDate, [3] = list of reasons for retraction
                        ###writer.writerow([OriginalPaperPubMedID_reason_date[1],OriginalPaperPubMedID_reason_date[3],abstract_text,OriginalPaperPubMedID_reason_date[2]])
                        writer.writerow([OriginalPaperPubMedID_reason_date[1],OriginalPaperPubMedID_reason_date[3],abstract_text])
                    
                    early_abstracts_found += 1
                    similar_abstracts.append(similar_abstract)
                    
                    with open('semantic_scholar_abstract30_early.csv','a',encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([similar_abstract])
                    
            time.sleep(1)
            
    print("\nlen(OriginalPaperPubMedID_reason_dates)",len(OriginalPaperPubMedID_reason_dates),"abstracts_found",retracted_abstracts_found,
          similar_abstracts_found,recent_abstracts_found, early_abstracts_found)
    
    print("\n",datetime.datetime.now())