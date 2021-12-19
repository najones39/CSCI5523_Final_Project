# _*_ coding=utf-8 _*_
import numpy as np
from sklearn import model_selection,svm,naive_bayes,neighbors,linear_model
from sklearn import metrics

data=np.load('leukemia_balanced_X.npy') #   breast/97-24481   ovarian/253-15154  colon/62(22-40)-2000
label=np.load('leukemia_balanced_y.npy')#cns/60-7129   leukemia/72-7129 prostate/136(59-77)-12600

def VSSRFE(n_genes):
    X=data
    y=label
    m, n_features = X.shape
    clf = svm.LinearSVC(penalty='l1',loss='squared_hinge',C=0.7,dual=False,
                        fit_intercept=True,random_state=1)
    count = n_features
    cut=n_features
    n_steps=400
    X_tmp=X
    while count > n_genes:
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
def model(n_genes):
    data=VSSRFE(n_genes)
    ls_acc=[];ls_mcc=[];ls_auc=[]
    skf=model_selection.StratifiedKFold(n_splits=5,shuffle=False,random_state=None)
    # clf=naive_bayes.GaussianNB()     # use NB as the classifier
    # clf=neighbors.NearestNeighbors(n_neighbors=7)   # use kNN as the classifier
    clf = linear_model.LogisticRegression(n_jobs=3, penalty='l2', solver='liblinear', C=19)   # use LR as the classifier
    # clf = svm.SVC(kernel='linear',C=3,probability=True)   # use SVM as the classifier
    for train_index, test_index in skf.split(data,label):
        clf.fit(data[train_index], label[train_index])
        acc=clf.score(data[test_index],label[test_index])
        ls_acc.append(acc)
        mcc=metrics.matthews_corrcoef(label[test_index],clf.predict(data[test_index]))
        ls_mcc.append(mcc)
        fpr, tpr, thresholds = metrics.roc_curve(label[test_index], (clf.predict_proba(data[test_index]))[:,1])
        auc=metrics.auc(fpr,tpr)
        ls_auc.append(auc)
    return np.mean(ls_acc), np.mean(ls_auc), np.mean(ls_mcc)

if __name__ == '__main__':
    res_acc=[];res_auc=[];res_mcc=[]
    for i in range(1, 33):
        ac,uc,mc=model(i)
        res_acc.append(ac)
        res_auc.append(uc)
        res_mcc.append(mc)
    print('acc:',res_acc)
    print('auc:',res_auc)
    print('mcc:',res_mcc)
    
    
    
    
    
    
    
    
    
    
    