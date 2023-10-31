import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, f1_score, accuracy_score, recall_score, precision_score

def confusion_matrix(testY, test_pred):
    TP, FP, TN, FN = 0, 0, 0, 0
    for (y,pred) in zip(testY, test_pred):
        if y == 1 and pred==1:
            TP+=1

        elif y==0 and pred==1:
            FP+=1

        elif y == 0 and pred==0:
            TN+=1
            
        elif y==1 and pred==0:
            FN+=1
    
    print('     y_true') 
    print('pred',[TP, FP],'\n    ',[FN,TN])

def plot_roc_curve(testY, test_pred, test_prob):    
    fpr, tpr, thresholds = roc_curve(testY, test_prob) # output 3개가 나오는데, 각 threshhold 마다의 fpr, tpr값 인듯
    
    test_f1 = f1_score(testY, test_pred)

    test_recall = recall_score(testY, test_pred)

    test_pre = precision_score(testY, test_pred)  

    test_acc = accuracy_score(testY, test_pred)
    
    plt.plot(fpr, tpr, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('test ROC : {}'.format(round(roc_auc_score(testY, test_prob),3)),fontsize=16)
    plt.figure(figsize=(6,6))
    plt.legend()
    plt.show()
    print('test_f1 score: ',test_f1,'\n')

    print('test_recall score: ',test_recall,'\n')

    print('test_pre score: ',test_pre,'\n')
    
    print('test acc score: ',test_acc, '\n')