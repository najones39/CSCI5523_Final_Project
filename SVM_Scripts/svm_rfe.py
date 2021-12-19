# _*_ coding=utf-8 _*_
import numpy as np
import sklearn.feature_selection as fs
from sklearn import model_selection,svm
from sklearn import metrics

data=np.load('/Users/Nick/Documents/Bioinformatics/Classes/CSCI_5523/Group_Project/leukemia_data.npy')
label=np.load('/Users/Nick/Documents/Bioinformatics/Classes/CSCI_5523/Group_Project/leukemia_label.npy')

label = label>0
def RFE(n_genes):
    rf = svm.SVC(kernel='linear',C=0.1,degree=3, tol=1e-4, cache_size=1000,
                 shrinking=True, probability=True,random_state=1,max_iter=-1)
    rfe=fs.RFE(rf,n_features_to_select=n_genes,step=2)
    new_data=rfe.fit_transform(data,label)
    return new_data

def model(n_genes):
    data=RFE(n_genes)
    ls_acc=[];ls_mcc=[];ls_auc=[]
    skf=model_selection.StratifiedKFold(n_splits=5,shuffle=False,random_state=None)
    fs = svm.SVC(kernel='linear', C=1.0,probability=True)
    for train_index, test_index in skf.split(data,label):
        fs.fit(data[train_index], label[train_index])
        acc=fs.score(data[test_index],label[test_index])
        ls_acc.append(acc)
        mcc=metrics.matthews_corrcoef(label[test_index],fs.predict(data[test_index]))
        ls_mcc.append(mcc)
        fpr, tpr, thresholds = metrics.roc_curve(label[test_index], (fs.predict_proba(data[test_index]))[:,1])
        auc=metrics.auc(fpr,tpr)
        ls_auc.append(auc)
    return np.mean(ls_acc), np.mean(ls_auc), np.mean(ls_mcc)

if __name__ == '__main__':
    res_acc=[];res_auc=[];res_mcc=[]
    for i in range(1, 129):
        ac,uc,mc=model(i)
        res_acc.append(ac)
        res_auc.append(uc)
        res_mcc.append(mc)
    print('acc:',res_acc)
    print('auc:',res_auc)
    print('mcc:',res_mcc)
