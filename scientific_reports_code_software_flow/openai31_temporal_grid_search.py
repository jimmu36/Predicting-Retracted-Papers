###mistral31_temporal_grid_search adds: 1) scaler.fit_transform to Xtest and splits out Xval/Yval from Xtrain/Ytrain and
###uses the validation datasets to set the SVM rbf kernel parameters C and gamma.
###openai30_temporal_grid_search follows from openai16_temporal_retracted_not_retracted_grid_search. It uses get_abstract30_temporal,
###which deletes papers with "retract" or with "withdrawn" in the abstract.
###The _temporal_retracted_not_retracted version makes sure the similar paper is in the early papers (before 2021 or earlier as OpenAI was
###generally made available by end of Sept '21) or later papers (2022 or later) year ranges.
###get_abstract16 uses from year 2001 - 2/15/25 and all subjects rather than the limitations in get_abstract14/15.
###get_abstract15 uses semantic scholar to get similar abstracts rather than openai in get_abstracts14. The goal will be to predict whether
###a paper is a retracted paper or not, rather than predicting if a retracted paper was due to error or fraud.
###get_abstract14 follows from get_abstract10. It tries to duplicate the dataset of the paper "Understanding and Predicting Retractions of 
###Published Work", which only uses subjects Health Sciences (HSC, 5,396 papers), Social Sciences (SOC, 2,651 papers), and Humanities 
###(HUM, 366 papers) from 2001 to 2019.
###openai_svm10 uses the reasons for the retraction as well, which will allow the use of the embeddings for multiple
###tests. All fraud/error reasons, focus (top 13fraud/6error reasons), focus 2 (top 6fraud/6error reasons), data, and result reasons.
###The default is all, but is passed in on the command line: python3.9 openai_svm10.py --reasons focus
###openai_svm6 uses retraction_error_fraud_embeddings5.csv, which are the correct embeddings.
###openai_svm4 follows from openai_svm2 rather than openai_svm3. Instead of doing the grid search, it just uses class = "balanced", and uses the defaults
###for C = 1 and gamma = 'scale'.
###openai_svm2 uses StandardScalar().fit_transform(X) before doing the SVM grid search
###openai_svm sets the majority class to 0 and the minority class to 1 to give the best indication if something can be learned from the dataset, so this
###version sets frauds to 0 and errors to 1. Last, openAI has been around longer, so to get more data, the recent date is pushed back to end Sept '21. Same as
###with retractions_error_fraud4.
###retractions_error_or_fraud4 uses test data with original publication dates after Sept '21, which is the last data that OpenAI's text-embedding-3-small 
###used in developing its embeddings. This prevents any of the test data from having already been seen by text-embedding-3-small, which
###could make it easier for the embeddings to reflect the specific test data.
import datetime
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, mean_squared_error, confusion_matrix, roc_auc_score, roc_curve, auc, average_precision_score
from sklearn.metrics import make_scorer, precision_recall_curve, auc, roc_curve
import csv
import pickle
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import argparse
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

#this class holds the information for each abstract
class Node:
    def __init__(self):
        self.embedding = []
        self.num_words = 0
        self.retracted_or_not_retracted = -1   ###retracted = 1, not-retracted = 0
        self.date = ''
    def __repr__(self):
        return f'Node({self.error_or_fraud}, {self.num_words}, {self.embedding})'
    
###def aucpr_score(y_true, y_scores):Ytest,y_pred
def aucpr_score(Ytest,y_pred):
    precision, recall, _ = precision_recall_curve(Ytest,y_pred)
    return auc(recall, precision)
    
def get_recent(date_this_line):
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
        return False

if __name__ == "__main__":
    print(datetime.datetime.now())
    early_papers = []
    ###with open('retraction_watch_embeddings30_early.csv', 'r') as file:
    with open('retraction_watch_embeddings30_early_no_citations.csv', 'r') as file:
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
    ###with open('semantic_scholar_embeddings30_early.csv', 'r') as file:
    with open('semantic_scholar_embeddings30_early_no_citations.csv', 'r') as file:
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
    ###with open('retraction_watch_embeddings30_recent.csv', 'r') as file:
    with open('retraction_watch_embeddings30_recent_no_citations.csv', 'r') as file:
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
    ###with open('semantic_scholar_embeddings30_recent.csv', 'r') as file:
    with open('semantic_scholar_embeddings30_recent_no_citations.csv', 'r') as file:
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
                
    print("\nlen early papers, recent papers, embeddings", len(early_papers), len(recent_papers), len(early_papers[0].embedding))

    #use the early papers, before text-embedding-3-small was trained, for training and the recent papers as test
    scaler = StandardScaler()
    early_embeddings = scaler.fit_transform(np.array([early_papers[i].embedding for i in range(len(early_papers))]))
    early_labels = np.array([early_papers[i].retracted_or_not_retracted for i in range(len(early_papers))])
    print("early_embeddings",early_embeddings.shape)
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
    
    Xtest = scaler.transform(np.array([recent_papers[i].embedding for i in range(len(recent_papers))]))
    Ytest = np.array([recent_papers[i].retracted_or_not_retracted for i in range(len(recent_papers))])
    print("len Xtest Ytest",len(Xtest),len(Ytest))
    
    aucpr_best = 0
    ###print("mistral: train with valid high citation papers, test with random papers ")
    ###QLoRA
    ###for C in [1., 10., 100.]:
    ###    for gamma in [1e-3, 1e-4, 1e-5]:
    ###IA3
    for C in [10., 100., 1000.]:
        for gamma in [1e-3, 1e-4, 1e-5]:
            print("C", C, "gamma", gamma,datetime.datetime.now())
            classifier = svm.SVC(kernel="rbf", class_weight = 'balanced', C=C, gamma=gamma, probability=True)
            classifier.fit(Xtrain, Ytrain)
            y_pred = classifier.predict_proba(Xval)[:,1]
            aucpr = average_precision_score(Yval, y_pred)
            print("validation: aucpr", aucpr)
            if aucpr > aucpr_best:
                aucpr_best = aucpr
                roc_auc = roc_auc_score(Yval,y_pred)
                thresholds = [i/100 for i in range(1,100)]
                f1_scores = [f1_score(Yval, (y_pred > t).astype(int)) for t in thresholds]  
                best_t = thresholds[np.argmax(f1_scores)]
                print("validation: aucpr_best",aucpr_best, "roc_auc",roc_auc,)
                print("validation: Best threshold:", best_t, "F1:", max(f1_scores))
                C_best = C
                gamma_best = gamma
                y_pred = classifier.predict_proba(Xtest)[:,1]
                aucpr_test = average_precision_score(Ytest, y_pred)
                roc_auc_test = roc_auc_score(Ytest,y_pred)
                f1_scores_test = [f1_score(Ytest, (y_pred > t).astype(int)) for t in thresholds]  
                best_t_test = thresholds[np.argmax(f1_scores)]
                print("test: aucpr_best",aucpr_test, "roc_auc",roc_auc_test)
                print("test: Best threshold:", best_t_test, "F1:", max(f1_scores_test))
    print("C_best",C_best,"gamma_best",gamma_best,datetime.datetime.now())
    

    