###mistral30_temporal_grid_search follows from sciber30_temporal_grid_search.
###scibert30_temporal_grid_search follows from openai30_temporal_grid_search.
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
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import pickle
import csv

#this class holds the information for each abstract
class Node:
    def __init__(self):
        self.embedding = []
        self.fraud_error = -1
        self.retracted_or_not_retracted = -1   ###retracted = 1, not-retracted = 0
        self.date = ''
    def __repr__(self):
        return f'Node({self.error_or_fraud}, {self.num_words}, {self.embedding})'
    
def get_indices(papers, len_indices, fraud_or_error):
    indices = np.sort(np.random.choice(len_indices, size=len_indices//10, replace=False))
    idx = -1
    indices_returned = []
    for i in range(len(papers)):
        if papers[i].fraud_error == fraud_or_error:
            idx += 1
            if idx in indices:
                indices_returned.append(i)
    return (np.sort(np.array(indices_returned)))

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
                fraud_reasons[line[0]] = 0
            elif line[2] == '0':
                error_reasons[line[0]] = 0
    return fraud_reasons, error_reasons
'''
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
                fraud_reasons[line[0]] = 0
            elif line[2] == '0':
                error_reasons[line[0]] = 0
    return fraud_reasons, error_reasons
'''
if __name__ == "__main__":
    print(datetime.datetime.now())
    ###Get the fraud and error reasons in the actual database. This is only used to drop the indeterminate reason abstracts.
    fraud_reasons, error_reasons = fraud_error_reasons('retraction_watch_2-15_count_no_blank_pubmedid16.csv')
    fraud_reasons_pred, error_reasons_pred = fraud_error_reasons('retraction_watch_2-15_count_no_blank_pubmedid16.csv')
    print("len(fraud_reasons) len(error_reasons)",len(fraud_reasons), len(error_reasons))
    reasons_this_line = []
    
    recent_papers = []
    recent_fraud = 0
    recent_error = 0
    with open('retraction_watch_abstract30_recent.csv', 'r', encoding="utf8") as file:
        for i, line in enumerate(file):
            if len(line) > 1:
                paper = Node()
                line_split = line.replace(';','').replace('+','',1).split(',')
                ###print(line_split[0],len(line_split[1].split('+')),line_split[1].split('+'))
                ###In retraction_watch_abstract30_recent.csv, 1 = fraud/0 = error. However, for analysis of the model, 1 needs to be the value
                ###of the smaller class, so that is inverted here as there are many more fraud cases in the Retraction Watch database than
                ###error cases.
                if int(line_split[0]) == 0:
                    paper.fraud_error = 1   ###1 = error
                    recent_error += 1
                    reasons = list(set(error_reasons) & set(line_split[1].split('+')))
                    for reason in reasons:
                        error_reasons[reason] += 1
                else:
                    paper.fraud_error = 0   ###0 = fraud
                    recent_fraud += 1
                    reasons = list(set(fraud_reasons) & set(line_split[1].split('+')))
                    for reason in reasons:
                        fraud_reasons[reason] += 1
                reasons_this_line.append(reasons)
                recent_papers.append(paper)
    num_retracted_abstracts = len(recent_papers)
    print("num_retracted_abstracts",num_retracted_abstracts, "recent_error", recent_error, "recent_fraud", recent_fraud)
    with open('retraction_watch_embeddings30_mistral_recent.csv', 'r') as file:
    ###with open('retraction_watch_embeddings30_mistral_lorabatch1_recent.csv', 'r') as file:
    ###with open('retraction_watch_embeddings30_lorabatch8_recent.csv', 'r') as file:
    ###with open('retraction_watch_embeddings30_lora_batch8_epochs5_recent.csv', 'r') as file:
    ###with open('retraction_watch_embeddings30_ia3_8batch_5epochs_recent.csv', 'r') as file:
        for i, line in enumerate(file):
            ###replace elements of the abstract with nothing so that it doesn't confuse ChatGPT
            line = line.replace('[','').replace(']','').replace("\n",'')
            if line != '':
                line_split = []
                line = line.split(',')
                for element in line:
                    line_split.append(float(element.strip().replace('"','').replace("'","")))
                recent_papers[i].embedding = line_split

    '''
    with open('semantic_scholar_embeddings30_mistral_recent.csv', 'r') as file:
    ###with open('retraction_watch_embeddings30_mistral_lorabatch1_recent.csv', 'r') as file:
    ###with open('semantic_scholar_embeddings30_lorabatch8_recent.csv', 'r') as file:
    ###with open('semantic_scholar_embeddings30_lora_batch8_epochs5_recent.csv', 'r') as file:
    ###with open('semantic_scholar_embeddings30_ia3_8batch_5epochs_recent.csv', 'r') as file:
    ###with open('semantic_scholar_embeddings30_mistral_ia3_recent.csv', 'r') as file:
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
    '''
    early_papers = []
    early_fraud = 0
    early_error = 0
    with open('retraction_watch_embeddings30_mistral_early.csv', 'r') as file:
    ###with open('retraction_watch_embeddings30_mistral_lorabatch1_early.csv', 'r') as file:
    ###with open('retraction_watch_embeddings30_lorabatch8_early.csv', 'r') as file:
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
                early_papers.append(paper)
    with open('retraction_watch_abstract30_early.csv', 'r') as file:
        for i, line in enumerate(file):
            if line != '':
                ###In retraction_watch_abstract30_early.csv, 1 = fraud/0 = error. However, for analysis of the model, 1 needs to be the value
                ###of the smaller class, so that is inverted here as there are many more fraud cases in the Retraction Watch database than
                ###error cases.
                if int(line[0]) == 1:
                    early_papers[i].fraud_error = 0   ###now 0 = fraud
                    early_fraud += 1
                else:
                    early_papers[i].fraud_error = 1   ###now 1 = error   
                    early_error += 1
    '''
    with open('semantic_scholar_embeddings30_mistral_early.csv', 'r') as file:
    ###with open('semantic_scholar_embeddings30_mistral_lorabatch1_early.csv', 'r') as file:
    ###with open('semantic_scholar_embeddings30_ia3_8batch_5epochs_early.csv', 'r') as file:
    ###with open('semantic_scholar_embeddings30_mistral_ia3_early.csv', 'r') as file:
    ###with open('semantic_scholar_embeddings30_lorabatch8_early.csv', 'r') as file:
    ###with open('semantic_scholar_embeddings30_lora_batch8_epochs5_early.csv', 'r') as file:
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
    '''
    print("\nlen early papers, recent papers, embeddings", len(early_papers), len(recent_papers), len(early_papers[0].embedding))
 
    #use the early papers, before text-embedding-3-small was trained, for training and the recent papers as test
    scaler = StandardScaler()
    early_embeddings = scaler.fit_transform(np.array([early_papers[i].embedding for i in range(len(early_papers))]))
    early_labels = np.array([early_papers[i].fraud_error for i in range(len(early_papers))])
    print("early_embeddings",early_embeddings.shape)
    print("early_labels",early_labels.shape)
    len_input = len(early_labels)
    np.random.seed(17)
    ###10% of input used for validation
    fraud_indices = get_indices(early_papers, early_fraud, 0)   ###0 for fraud, 1 for error
    error_indices = get_indices(early_papers, early_error, 1)
    indices = np.concatenate((fraud_indices,error_indices))
    Xtrain = scaler.fit_transform(np.array([early_papers[i].embedding for i in range(len(early_papers)) if i not in indices]))
    Ytrain = np.array([early_papers[i].fraud_error for i in range(len(early_papers)) if i not in indices])
    print("len Xtrain Ytrain",len(Xtrain),len(Ytrain))
    Xval = scaler.transform(np.array([early_papers[i].embedding for i in indices]))
    Yval = np.array([early_papers[i].fraud_error for i in indices])

    Xtest = scaler.transform(np.array([recent_papers[i].embedding for i in range(len(recent_papers))]))
    Ytest = np.array([recent_papers[i].fraud_error for i in range(len(recent_papers))])
    aucpr_best = 0
    
    C = 1.
    gamma = 0.0001
    print("C", C, "gamma", gamma,datetime.datetime.now())
    classifier = svm.SVC(kernel="rbf", class_weight = 'balanced', C=C, gamma=gamma, probability=True)
    classifier.fit(Xtrain, Ytrain)
    y_pred = classifier.predict_proba(Xtest)[:,1]
    aucpr = average_precision_score(Ytest, y_pred)
    roc_auc = roc_auc_score(Ytest,y_pred)
    thresholds = [i/100 for i in range(1,100)]
    f1_scores = [f1_score(Ytest, (y_pred > t).astype(int)) for t in thresholds]  
    best_t = thresholds[np.argmax(f1_scores)]
    print("aucpr",aucpr, "roc_auc",roc_auc)
    print("Best threshold:", best_t, "F1:", max(f1_scores))
    print(datetime.datetime.now())
    ###the following gives the binary prediction
    y_pred_binary = (y_pred >= best_t).astype(int)
    correct_fraud = 0
    correct_error = 0
    ###for i, pred in enumerate(y_pred_binary):    
    for i in range(num_retracted_abstracts):
        pred = y_pred_binary[i]
        if pred == Ytest[i]:   ###prediction matches valid=0 or retracted=1
            if pred == 1:   ###1 = retracted
                if recent_papers[i].fraud_error == 0:   ###0=fraud
                    correct_fraud += 1
                else:   ###1 = error
                    correct_error += 1
    print("fraud", correct_fraud, " / ", recent_fraud)
    print("error", correct_error, " / ", recent_error)
    
    print("precision", precision_score(Ytest, y_pred_binary))
    for i, pred in enumerate(y_pred_binary):
        if pred == 0:   ###0 = fraud
            for reason in reasons_this_line[i]:
                if reason in fraud_reasons_pred:
                    fraud_reasons_pred[reason] += 1
        else:   ###1 = error
            for reason in reasons_this_line[i]:
                if reason in error_reasons_pred:
                    error_reasons_pred[reason] += 1
    print("frauds")
    sorted_fraud_reasons_pred = dict(sorted(fraud_reasons_pred.items(), key=lambda item: item[1], reverse=True))
    for reason in sorted_fraud_reasons_pred:
        if fraud_reasons[reason] == 0:
            print(reason, sorted_fraud_reasons_pred[reason], " / ", fraud_reasons[reason])
        else:
            print(reason, sorted_fraud_reasons_pred[reason], " / ", fraud_reasons[reason], " = ", sorted_fraud_reasons_pred[reason]/fraud_reasons[reason])
    print("errors")
    sorted_error_reasons_pred = dict(sorted(error_reasons_pred.items(), key=lambda item: item[1], reverse=True))
    for reason in sorted_error_reasons_pred:
        if error_reasons[reason] == 0:
            print(reason, sorted_error_reasons_pred[reason], " / ", error_reasons[reason])
        else:
            print(reason, sorted_error_reasons_pred[reason], " / ", error_reasons[reason], " = ", sorted_error_reasons_pred[reason]/error_reasons[reason])
    

    print(datetime.datetime.now())
