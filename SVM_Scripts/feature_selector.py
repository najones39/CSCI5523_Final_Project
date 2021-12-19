# _*_ coding=utf-8 _*_
import numpy as np
import sklearn.feature_selection as fs
from sklearn import model_selection,svm
from sklearn import metrics
import time

data=np.load('leukemia_balanced_X.npy') # breast/97-24481  ovarian/253-15154  colon/62(22-40)-2000
label=np.load('leukemia_balanced_y.npy')# cns/60-7129  leukemia/72-7129 prostate/136(59-77)-12600

def rfe(threshold):
    rf = svm.SVC(kernel='linear', C=0.5, probability=True,random_state=1)
    rfe=fs.RFE(rf,n_features_to_select=threshold,step=1)
    new_data=rfe.fit_transform(data,label)
    return new_data
def vssrfe(n_selected_features):
    X=data
    y=label
    m, n_features = X.shape
    clf = svm.SVC(kernel='linear', C=0.1, probability=True,  random_state=1)
    count = n_features
    cut=n_features
    n_steps=400
    X_tmp=X
    while count > n_selected_features:
          count = count - n_steps
          if cut / count==2 and n_steps>1:
             cut=count
             n_steps=n_steps/2
          clf.fit(X_tmp, y)
          coef=[abs(e) for e in (clf.coef_)[0]]
          coef_sorted=np.argsort(coef)[::-1]
          coef_eliminated=coef_sorted[:count]
          X_tmp = X_tmp[:, coef_eliminated]
    return X_tmp
def model(n_selected_features):
    # data=rfe(n_selected_features)       # use SVM-RFE as the feature selector
    data = vssrfe(n_selected_features)    # use SVM-VSSRFE as the feature selector
    ls_acc=[];ls_mcc=[];ls_auc=[]
    skf=model_selection.StratifiedKFold(n_splits=5,shuffle=False,random_state=None)
    fs = svm.SVC(kernel='linear',probability=True)
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
    t0=time.time()
    for i in range(1, 129):
        ac,uc,mc=model(i)
        res_acc.append(ac)
        res_auc.append(uc)
        res_mcc.append(mc)
    t1 = time.time()
    print('acc:',res_acc)
    print('auc:',res_auc)
    print('mcc:',res_mcc)
    print(t1-t0)

    # Evaluate the time consumption

    # t0 = time.time()
    # ac, uc, mc = model(4)
    # t1 = time.time()
    # t=t1-t0
    # print ac,uc,mc,t
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    