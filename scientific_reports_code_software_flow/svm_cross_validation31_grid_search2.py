###svm_cross_validation30_grid_search uses get_abstract30/get_embeddings30, which deletes papers with "retract" or with "withdrawn" in the abstract.
###svm_cross_validation21 uses the values of C and gamma from the grid search to generate the model for each fold of the cross validation. 
###Linear SVM kernels did not perform as well as rbf SVM kernels, so are commented out here.
import datetime
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, mean_squared_error, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.metrics import make_scorer, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import svm
from sklearn.model_selection import KFold
import argparse
import pickle
from sklearn.pipeline import Pipeline
import pandas as pd

#this node holds the information for each abstract
class Node:
    def __init__(self):
        self.embedding = []
        self.num_words = 0
        self.retracted_not_retracted = -1
        self.date = ''
    def __repr__(self):
        return f'Node({self.error_or_fraud}, {self.num_words}, {self.embedding})'

###This custom scorer uses optimizing AUC-PR as the criterion for the grid search.
def aucpr_score(Ytest,y_pred):
    precision, recall, _ = precision_recall_curve(Ytest,y_pred)
    return auc(recall, precision)

if __name__ == "__main__":
    print(datetime.datetime.now())
    retracted_papers = []
    valid_papers = []
    ###test = '_missing_top_10words'
    ###test = '_missing_top_25words'
    ###test = '_meta_llama3.1_missing_top_10words'
    ###test = '_meta_llama3.1_missing_top_25words'
    ###test = '_biosentvec'
    ###test = '_scibert'
    ###test = '_meta_llama3.1'
    ###test = '_google_gemini-001'
    ###test = '_voyageai'
    test = ''
    print('test', test)
    ###The embeddings are retrieved from the appropriate intermediate file and converted to floating numbers. Since these are retracted papers
    ###the target value is set to 1 (class=1 is retracted.)
    with open('retraction_watch_embeddings30_early.csv', 'r') as file:
    ###with open('retraction_embeddings30'+test+'.csv', 'r') as file:
        for i, line in enumerate(file):
            ###replace elements of the abstract with nothing so that it doesn't confuse ChatGPT
            line = line.replace('[','').replace(']','').replace("\n",'').replace('"','').replace('ContentEmbedding(values=','').replace(', statistics=None)','').replace('embedding=','')
            if line != '':
                line_split = line.split(',')  
                paper = Node()
                paper.retracted_not_retracted = 1 ###1 for retracted, 0 for not-retracted
                paper.embedding = [float(line_split[j]) for j in range(len(line_split))]
                retracted_papers.append(paper)
    with open('retraction_watch_embeddings30_recent.csv', 'r') as file:
    ###with open('retraction_embeddings30'+test+'.csv', 'r') as file:
        for i, line in enumerate(file):
            ###replace elements of the abstract with nothing so that it doesn't confuse ChatGPT
            line = line.replace('[','').replace(']','').replace("\n",'').replace('"','').replace('ContentEmbedding(values=','').replace(', statistics=None)','').replace('embedding=','')
            if line != '':
                line_split = line.split(',')  
                paper = Node()
                paper.retracted_not_retracted = 1 ###1 for retracted, 0 for not-retracted
                paper.embedding = [float(line_split[j]) for j in range(len(line_split))]
                retracted_papers.append(paper) 
    print("len(retracted_papers)",len(retracted_papers))
    ###The embeddings are retrieved from the appropriate intermediate file and converted to floating numbers. Since these are valid papers
    ###the target value is set to 0 (class=0 is valid.)
    with open('semantic_scholar_embeddings30_early.csv', 'r') as file:
    ###with open('semantic_scholar_embeddings30'+test+'.csv', 'r') as file:
        for i, line in enumerate(file):
            line = line.replace('[','').replace(']','').replace("\n",'').replace('"','').replace('ContentEmbedding(values=','').replace(', statistics=None)','').replace('embedding=','')
            if line != '':
                line_split = line.split(',')
                paper = Node()
                paper.retracted_not_retracted = 0 ###1 for retracted, 0 for not-retracted
                paper.embedding = [float(line_split[j]) for j in range(len(line_split))]
                valid_papers.append(paper)
    with open('semantic_scholar_embeddings30_recent.csv', 'r') as file:
    ###with open('semantic_scholar_embeddings30'+test+'.csv', 'r') as file:
        for i, line in enumerate(file):
            line = line.replace('[','').replace(']','').replace("\n",'').replace('"','').replace('ContentEmbedding(values=','').replace(', statistics=None)','').replace('embedding=','')
            if line != '':
                line_split = line.split(',')
                paper = Node()
                paper.retracted_not_retracted = 0 ###1 for retracted, 0 for not-retracted
                paper.embedding = [float(line_split[j]) for j in range(len(line_split))]
                valid_papers.append(paper)
    print("len(valid_papers)",len(valid_papers))

    ###Set up the numpy arrays for using the C and gamma values found for the rbf kernel in the grid search.
    retracted_X = np.array([retracted_papers[i].embedding for i in range(len(retracted_papers))])
    retracted_Y = np.array([retracted_papers[i].retracted_not_retracted for i in range(len(retracted_papers))])
    valid_X = np.array([valid_papers[i].embedding for i in range(len(valid_papers))])
    valid_Y = np.array([valid_papers[i].retracted_not_retracted for i in range(len(valid_papers))])
 
    AUCPR = []
    folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=17)
    ###C_choices = [100., 1000., 10000.]
    ###gamma_choices = [0.1, 1., 10.]
    C_choices = [100., 1000., 10000.]
    gamma_choices = [0.001, 0.0001, 0.00001]
    scaler = StandardScaler()
    fold_iteration = -1
    for fold, (train_idx, test_idx) in enumerate(folds.split(retracted_X, retracted_Y), 1):
        fold_iteration += 1
        for C in C_choices:
            for gamma in gamma_choices:
                print("fold iteration",fold_iteration,"C", C, "gamma", gamma,datetime.datetime.now())
            
                Xtrain = scaler.fit_transform(np.array([retracted_X[i] for i in train_idx]+[valid_X[j] for j in train_idx]))
                Ytrain = np.array([retracted_Y[i] for i in train_idx]+[valid_Y[j] for j in train_idx])
                ###print("Xtrain",Xtrain.shape)
                ###print("Ytrain",Ytrain.shape)
                Xtest = scaler.transform(np.array([retracted_X[i] for i in test_idx]+[valid_X[j] for j in test_idx]))
                Ytest = np.array([retracted_Y[i] for i in test_idx]+[valid_Y[j] for j in test_idx])
                classifier = svm.SVC(kernel="rbf", class_weight = 'balanced', C=C, gamma=gamma, probability=True)
                classifier.fit(Xtrain, Ytrain)
                y_pred = classifier.predict_proba(Xtest)[:,1]
                aucpr = average_precision_score(Ytest, y_pred)
                AUCPR.append(aucpr/4)
                roc_auc = roc_auc_score(Ytest,y_pred)
                thresholds = [i/100 for i in range(1,100)]
                f1_scores = [f1_score(Ytest, (y_pred > t).astype(int)) for t in thresholds]  
                best_t = thresholds[np.argmax(f1_scores)]
                print("roc_auc",roc_auc,"aucpr",aucpr)
                ###print("Best threshold:", best_t, "F1:", max(f1_scores))
    for i in range(len(AUCPR)):
        print(AUCPR[i])
    print()
    for i in range(9):
        print('C', C_choices[i//3], 'gamma', gamma_choices[i%3])
        print('average aucpr'+str(i), AUCPR[i]+AUCPR[i+9]+AUCPR[i+18]+AUCPR[i+27])
    print(datetime.datetime.now()) 
    