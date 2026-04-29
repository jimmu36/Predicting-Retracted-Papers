###svm_cross_validation30_grid_search uses get_abstract30/get_embeddings30, which deletes papers with "retract" or with "withdrawn" in the abstract.
###svm_cross_validation21 uses the values of C and gamma from the grid search to generate the model for each fold of the cross validation. 
###Linear SVM kernels did not perform as well as rbf SVM kernels, so are commented out here.
import datetime
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, mean_squared_error, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.metrics import make_scorer, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

#this node holds the information for each abstract
class Node:
    def __init__(self):
        self.embedding = []
        self.abstract = ''
        self.retracted_not_retracted = -1
        self.date = ''
    def __repr__(self):
        return f'Node({self.error_or_fraud}, {self.num_words}, {self.embedding})'

if __name__ == "__main__":
    print(datetime.datetime.now())
    retracted_papers = []
    valid_papers = []
    abstracts = []
    valid_retracted = []   ###0 for valid, 1 for retracted
    with open('retraction_watch_abstract30_early.csv', 'r', encoding="utf8") as file:
        for i, line in enumerate(file):
            ###if i % 1000 == 0:
            ###    print(i,datetime.datetime.now())
            ###print(i,end=".")
            if len(line) > 1:
                abstract = line[line.find('Abstract')+9:]
                abstracts.append(abstract)
                valid_retracted.append(1) 
                paper = Node()
                paper.abstract = abstract
                paper.retracted_not_retracted = 1 ###1 for retracted, 0 for not-retracted
                retracted_papers.append(paper)

    with open('semantic_scholar_abstract30_early.csv', 'r', encoding="utf8") as file:
        for i, line in enumerate(file):
            ###if i % 1000 == 0:
            ###    print(i,datetime.datetime.now())
            ###print(i,end=".")
            if len(line) > 0:
                abstract = line.replace('\n','').replace('"','')
                abstracts.append(abstract)
                valid_retracted.append(0)
                paper = Node()
                paper.retracted_not_retracted = 0 ###1 for retracted, 0 for not-retracted
                paper.abstract = abstract
                valid_papers.append(paper)
                
    recent_abstracts = []
    recent_valid_retracted = []
    with open('retraction_watch_abstract30_recent.csv', 'r', encoding="utf8") as file:
        for i, line in enumerate(file):
            ###if i % 1000 == 0:
            ###    print(i,datetime.datetime.now())
            ###print(i,end=".")
            if len(line) > 1:
                abstract = line[line.find('Abstract')+9:]
                abstract = abstract.replace('\n','').replace('"','')
                recent_abstracts.append(abstract)
                recent_valid_retracted.append(1)  
                paper = Node()
                paper.retracted_not_retracted = 1 ###1 for retracted, 0 for not-retracted
                paper.abstract = abstract
                retracted_papers.append(paper) 
    
    with open('semantic_scholar_abstract30_recent.csv', 'r', encoding="utf8") as file:
        for i, line in enumerate(file):
            ###if i % 1000 == 0:
            ###    print(i,datetime.datetime.now())
            ###print(i,end=".")
            if len(line) > 0:
                abstract = line.replace('\n','').replace('"','')
                recent_abstracts.append(abstract)
                recent_valid_retracted.append(0)
                paper = Node()
                paper.retracted_not_retracted = 0 ###1 for retracted, 0 for not-retracted
                paper.abstract = abstract
                valid_papers.append(paper)
    print("len(retracted_papers)",len(retracted_papers))
    print("len(valid_papers)",len(valid_papers))

    ###Set up the numpy arrays for using the C and gamma values found for the rbf kernel in the grid search.
    retracted_X = np.array([retracted_papers[i].abstract for i in range(len(retracted_papers))])
    retracted_Y = np.array([retracted_papers[i].retracted_not_retracted for i in range(len(retracted_papers))])
    valid_X = np.array([valid_papers[i].abstract for i in range(len(valid_papers))])
    valid_Y = np.array([valid_papers[i].retracted_not_retracted for i in range(len(valid_papers))])
 
    AUCPR = []
    folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=17)
    C_choices = [1., 10., 100.]
    gamma_choices = [0.1, 1., 10.]
    for fold, (train_idx, test_idx) in enumerate(folds.split(retracted_X, retracted_Y), 1):
        ###for C in [1.]:
        ###    for gamma in [10.]:
        for C in C_choices:
            for gamma in gamma_choices:
                print("C", C, "gamma", gamma,datetime.datetime.now())
                Xtrain = [retracted_X[i] for i in train_idx]+[valid_X[j] for j in train_idx]
                Ytrain = [retracted_Y[i] for i in train_idx]+[valid_Y[j] for j in train_idx]
                print("Xtrain",len(Xtrain))
                print("Ytrain",len(Ytrain))
                Xtest = [retracted_X[i] for i in test_idx]+[valid_X[j] for j in test_idx]
                Ytest = [retracted_Y[i] for i in test_idx]+[valid_Y[j] for j in test_idx]
                model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), stop_words="english", min_df=11)),
                ###model = Pipeline([("tfidf", TfidfVectorizer(max_features=50000, max_features=50000, ngram_range=(1,2), stop_words="english")),
                ###                  ("clf", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.01, force_row_wise=True))])
                               ("clf", svm.SVC(kernel="rbf", class_weight = 'balanced', C=C, gamma=gamma, probability=True))])
                model.fit(Xtrain, Ytrain)
                y_pred = model.predict_proba(Xtest)[:,1]
                aucpr = average_precision_score(Ytest, y_pred)
                AUCPR.append(aucpr/4)
                roc_auc = roc_auc_score(Ytest,y_pred)
                thresholds = [i/100 for i in range(1,100)]
                f1_scores = [f1_score(Ytest, (y_pred > t).astype(int)) for t in thresholds]  
                best_t = thresholds[np.argmax(f1_scores)]
                print("aucpr",aucpr, "roc_auc",roc_auc)
                print("Best threshold:", best_t, "F1:", max(f1_scores))
    for i in range(len(AUCPR)):
        print(i, AUCPR[i])
    for i in range(9):
        print('C', C_choices[i//3], 'gamma', gamma_choices[i%3])
        print('average aucpr'+str(i), AUCPR[i]+AUCPR[i+9],AUCPR[i+18]+AUCPR[i+27])
    print(datetime.datetime.now()) 
    