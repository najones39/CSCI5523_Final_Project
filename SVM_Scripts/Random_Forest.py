# _*_ coding=utf-8 _*_
import numpy as np
from sklearn import model_selection,svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

data=np.load('leukemia_balanced_X.npy')
label=np.load('leukemia_balanced_y.npy')-1

def rf(n_genes):
    rf=RandomForestClassifier(n_estimators=100,max_depth=5,n_jobs=3,oob_score=False,bootstrap=True,
                                       random_state=1)#
    rf.fit(data, label)
    fea_imp = rf.feature_importances_
    index = (np.argsort(fea_imp))[(len(fea_imp)-n_genes):]
    ls = list(range(0, len(fea_imp)))
    for i in range(len(fea_imp)):
        if i not in index:
            ls[i] = False
        else:
            ls[i] = True
    newdata = data[:, np.array(ls)]
    return newdata

def model(threshold):
    data=rf(threshold)
    ls_acc=[];ls_mcc=[];ls_auc=[]
    skf=model_selection.StratifiedKFold(n_splits=5,shuffle=False,random_state=None)
    fs = svm.SVC(kernel='linear',C=1.0,probability=True)
    for train_index, test_index in skf.split(data,label):
        fs.fit(data[train_index], label[train_index])
        acc=fs.score(data[test_index],label[test_index])
        ls_acc.append(acc)
        mcc=metrics.matthews_corrcoef(label[test_index],fs.predict(data[test_index]))
        ls_mcc.append(mcc)
        #had to use abs() to get auc to show up at all...not sure if that actually fixed the issue
        fpr, tpr, thresholds = metrics.roc_curve(abs(label[test_index]), (fs.predict_proba(data[test_index]))[:,1])  
        auc=metrics.auc(fpr,tpr)
        ls_auc.append(auc)
    return np.mean(ls_acc), np.mean(ls_auc), np.mean(ls_mcc) #


if __name__ == '__main__':
    res_acc = [];
    res_auc = [];
    res_mcc = []
    for i in range(1, 129):
        ac, uc, mc = model(i)  #
        res_acc.append(ac)
        res_auc.append(uc)
        res_mcc.append(mc)
    print('acc:', res_acc)
    print('auc:', res_auc)
    print('mcc:', res_mcc)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    