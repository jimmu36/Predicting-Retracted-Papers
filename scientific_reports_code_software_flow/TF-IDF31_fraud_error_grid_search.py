###TF-IDF31_fraud_error_temporal_grid_search adds: 1) scaler.fit_transform to Xtest and splits out Xval/Yval from Xtrain/Ytrain and
###uses the validation datasets to set the SVM rbf kernel parameters C and gamma.
import datetime
import csv
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, mean_squared_error, confusion_matrix, roc_auc_score, roc_curve, auc, average_precision_score
from sklearn.metrics import make_scorer, precision_recall_curve, auc, roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import svm
    
def get_indices(papers, len_indices, fraud_or_error):
    indices = np.sort(np.random.choice(len_indices, size=len_indices//10, replace=False))
    idx = -1
    indices_returned = []
    for i in range(len(papers)):
        if papers[i] == fraud_or_error:
            idx += 1
            if idx in indices:
                indices_returned.append(i)
    return (np.sort(np.array(indices_returned)))

if __name__ == "__main__":
    print(datetime.datetime.now())
    ###Term Frequency - Inverse Document Frequency
    ###model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), stop_words="english")),
    ###                  ("clf", LogisticRegression(max_iter=1000))])
    abstracts = []
    fraud_error = []   ###0 for fraud, 1 for error
    early_fraud = 0
    early_error = 0
    with open('retraction_watch_abstract30_early.csv', 'r', encoding="utf8") as file:
        for i, line in enumerate(file):
            if i % 1000 == 0:
                print(i,datetime.datetime.now())
            ###print(i,end=".")
            if len(line) > 1:
                abstract = line[line.find('Abstract')+9:]
                abstracts.append(abstract)
                ###In retraction_watch_abstract30_early.csv, 1 = fraud/0 = error. However, for analysis of the model, 1 needs to be the value
                ###of the smaller class, so that is inverted here as there are many more fraud cases in the Retraction Watch database than
                ###error cases.
                if int(line[0]) == 1:
                    fraud_error.append(0)   ###now 0 = fraud
                    early_fraud += 1
                else:
                    fraud_error.append(1)   ###now 1 = error   
                    early_error += 1
    
    recent_abstracts = []
    recent_fraud_error = []
    recent_fraud = 0
    recent_error = 0
    with open('retraction_watch_abstract30_recent.csv', 'r', encoding="utf8") as file:
        for i, line in enumerate(file):
            if i % 1000 == 0:
                print(i,datetime.datetime.now())
            ###print(i,end=".")
            if len(line) > 1:
                abstract = line[line.find('Abstract')+9:]
                abstract = abstract.replace('\n','').replace('"','')
                recent_abstracts.append(abstract)
                ###In retraction_watch_abstract30_recent.csv, 1 = fraud/0 = error. However, for analysis of the model, 1 needs to be the value
                ###of the smaller class, so that is inverted here as there are many more fraud cases in the Retraction Watch database than
                ###error cases.
                if int(line[0]) == 1:
                    recent_fraud_error.append(0)   ###now 0 = fraud
                    recent_fraud += 1
                else:
                    recent_fraud_error.append(1)  ###now 1 = error
                    recent_error += 1
    print("\nlen early papers, recent papers, embeddings", len(abstracts), len(recent_abstracts))
    print("len early fraud/error, len recent fraud/error", early_fraud, early_error, recent_fraud, recent_error)

    np.random.seed(17)
    ###10% of input used for validation
    fraud_indices = get_indices(fraud_error, early_fraud, 0)   ###0 for fraud, 1 for error
    error_indices = get_indices(fraud_error, early_error, 1)
    indices = np.concatenate((fraud_indices,error_indices))
    Xtrain = np.array([abstracts[i] for i in range(len(fraud_error)) if i not in indices])
    Ytrain = np.array([fraud_error[i] for i in range(len(fraud_error)) if i not in indices])
    print("len Xtrain Ytrain",len(Xtrain),len(Ytrain))
    Xval = np.array([abstracts[i] for i in indices])
    Yval = np.array([fraud_error[i] for i in indices])
    print("len Xval Yval",len(Xval),len(Yval))
    Xtest = np.array([recent_abstracts[i] for i in range(len(recent_abstracts))])
    Ytest = np.array([recent_fraud_error[i] for i in range(len(recent_abstracts))])
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
    for C in [100, 1e3, 1e4]:
        for gamma in [1e-3, 1e-4, 1e-5]:
            print("C", C, "gamma", gamma,datetime.datetime.now())
            classifier = svm.SVC(kernel="rbf", class_weight = 'balanced', C=C, gamma=gamma, probability=True)
            classifier.fit(Xtrain, Ytrain)
            y_pred = classifier.predict_proba(Xval)[:,1]
            aucpr = average_precision_score(Yval, y_pred)
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
    ###print(df_top_terms.head(20))
    print("rbf: C_best",C_best,"gamma_best",gamma_best,datetime.datetime.now())
    
