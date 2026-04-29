import csv
import datetime
import time
from sklearn.metrics import make_scorer, precision_recall_curve, auc, roc_curve, average_precision_score, roc_auc_score
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print(datetime.datetime.now())
    y_pred = []
    Ytest = []
    ###with open('semantic_scholar_zero_shot30.csv', 'r') as f:
    with open('semantic_scholar_few_shot30.csv', 'r') as f:
        for i, line in enumerate(f):
            if len(line) > 0:
                y_pred.append(float(line.strip()))
                Ytest.append(0)
            else:
                print("len semantic_scholar line = 0", i)

    ###with open('retraction_watch_zero_shot30.csv', 'r') as f:
    with open('retraction_watch_few_shot30.csv', 'r') as f:
        for i, line in enumerate(f):
            if len(line) > 0:
                y_pred.append(float(line.strip()))
                Ytest.append(1)
            else:
                print("len retraction_watch line = 0", i)
    roc_auc = roc_auc_score(Ytest,y_pred)
    aucpr = average_precision_score(Ytest,y_pred)
    print("roc_auc",roc_auc,"aucpr",aucpr)
    '''
    plt.figure()
    label = 'Zero-shot AUC-ROC'
    roc_auc = roc_auc_score(Ytest,y_pred)
    print("rbf kernel: roc_auc",roc_auc)
    fpr, tpr, _ = roc_curve(Ytest, y_pred)
    roc_auc = auc(fpr, tpr)
    ###plt.plot(fpr, tpr, label=f'{test_label_dict[test]} = {roc_auc:.2f}', color='black', linestyle=linestyles_dict[test], marker=marker_dict[test])
    line, = plt.plot(fpr, tpr, label=f'{label} = {roc_auc:.2f}', color='black')
    line.set_dashes([1,1])
    
    aucpr = average_precision_score(Ytest,y_pred)
    print("aucpr",aucpr)
    precision, recall, _ = precision_recall_curve(Ytest, y_pred)
    auc_pr = auc(recall, precision)
    plt.figure()
    label = 'Zero-shot AUC-PR'
    line, = plt.plot(recall, precision, label=f'{label} = {auc_pr:.2f}', color='black')
    line.set_dashes([6,3])
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Area Under Curve-Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Random guess line
    plt.grid(True)
    plt.show()
    '''
    
    print("\n",datetime.datetime.now())