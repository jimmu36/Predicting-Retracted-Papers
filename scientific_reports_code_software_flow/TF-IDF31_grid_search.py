import datetime
import csv
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, mean_squared_error, confusion_matrix, roc_auc_score, roc_curve, auc, average_precision_score
from sklearn.metrics import make_scorer, precision_recall_curve, auc, roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import svm
import lightgbm as lgb
import pandas as pd

if __name__ == "__main__":
    print(datetime.datetime.now())
    ###Term Frequency - Inverse Document Frequency
    ###model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), stop_words="english")),
    ###                  ("clf", LogisticRegression(max_iter=1000))])
    abstracts = []
    valid_retracted = []   ###0 for valid, 1 for retracted
    with open('retraction_watch_abstract30_early.csv', 'r', encoding="utf8") as file:
        for i, line in enumerate(file):
            if i % 1000 == 0:
                print(i,datetime.datetime.now())
            ###print(i,end=".")
            if len(line) > 1:
                abstract = line[line.find('Abstract')+9:]
                abstracts.append(abstract)
                valid_retracted.append(1)
    
    with open('semantic_scholar_abstract30_early.csv', 'r', encoding="utf8") as file:
        for i, line in enumerate(file):
            if i % 1000 == 0:
                print(i,datetime.datetime.now())
            ###print(i,end=".")
            if len(line) > 0:
                abstract = line.replace('\n','').replace('"','')
                abstracts.append(abstract)
                valid_retracted.append(0)
    
    recent_abstracts = []
    recent_valid_retracted = []
    with open('retraction_watch_abstract30_recent.csv', 'r', encoding="utf8") as file:
        for i, line in enumerate(file):
            if i % 1000 == 0:
                print(i,datetime.datetime.now())
            ###print(i,end=".")
            if len(line) > 1:
                abstract = line[line.find('Abstract')+9:]
                abstract = abstract.replace('\n','').replace('"','')
                recent_abstracts.append(abstract)
                recent_valid_retracted.append(1)
    
    with open('semantic_scholar_abstract30_recent.csv', 'r', encoding="utf8") as file:
        for i, line in enumerate(file):
            if i % 1000 == 0:
                print(i,datetime.datetime.now())
            ###print(i,end=".")
            if len(line) > 0:
                abstract = line.replace('\n','').replace('"','')
                recent_abstracts.append(abstract)
                recent_valid_retracted.append(0)
    print("len abstracts, recent_abstract",len(abstracts),len(recent_abstracts))
    len_input = len(valid_retracted)
    np.random.seed(17)
    ###10% of input used for validation, but requires /2 additional as both retracted and valid papers are included,
    ###and they have to be put into validation together in pairs to not introduce bias
    indices_first_half = np.sort(np.random.choice(len_input//2, size=len_input//20, replace=False))
    ###print(indices_first_half)
    indices = np.concatenate((indices_first_half,np.array([i+len_input//2 for i in indices_first_half])))
    Xtrain = np.array([abstracts[i] for i in range(len(valid_retracted)) if i not in indices])
    Ytrain = np.array([valid_retracted[i] for i in range(len(valid_retracted)) if i not in indices])
    print("len Xtrain Ytrain",len(Xtrain),len(Ytrain))
    Xval = np.array([abstracts[i] for i in indices])
    Yval = np.array([valid_retracted[i] for i in indices])
    print("len Xval Yval",len(Xval),len(Yval))
    Xtest = np.array([recent_abstracts[i] for i in range(len(recent_abstracts))])
    Ytest = np.array([recent_valid_retracted[i] for i in range(len(recent_abstracts))])
    print("len Xtest Ytest",len(Xtest),len(Ytest))
    ###The following requires a minimum of 10 occurances of a word across all documents.
    embed = TfidfVectorizer(ngram_range=(1,2), stop_words="english", min_df=11)
    Xtrain = embed.fit_transform(Xtrain)
    print("Xtrain.shape",Xtrain.shape)
    Xval = embed.transform(Xval)
    print("Xval.shape",Xval.shape)
    Xtest = embed.transform(Xtest)
    print("Xtest.shape",Xtest.shape)
    
    aucpr_best = 0
    ###for C in [1e-2, 1e-1, 1, 10, 100, 1e3]:
    ###    for gamma in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
    ###for C in [10, 100, 1e3]:
    ###    for gamma in [1e-4, 1e-3, 1e-2, 0.1]:
    ###for C in [1, 10, 100]:
    ###    for gamma in [1e-2, 0.1, 1]:
    for C in [10, 100, 1000]:
        for gamma in [0.1, 1., 10.]:
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
                print("validation: roc_auc, best threshold",roc_auc, best_t, "F1:", max(f1_scores))
                C_best = C
                gamma_best = gamma
                y_pred = classifier.predict_proba(Xtest)[:,1]
                aucpr_test = average_precision_score(Ytest, y_pred)
                roc_auc_test = roc_auc_score(Ytest,y_pred)
                f1_scores_test = [f1_score(Ytest, (y_pred > t).astype(int)) for t in thresholds]  
                best_t_test = thresholds[np.argmax(f1_scores)]
                print("test: aucpr_best", aucpr_test, "roc_auc",roc_auc_test)
                print("test: Best threshold:", best_t_test, "F1:", max(f1_scores_test))
    ###print(df_top_terms.head(20))
    print("rbf: C_best",C_best,"gamma_best",gamma_best,datetime.datetime.now())

