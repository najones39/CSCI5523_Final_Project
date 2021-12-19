# _*_ coding=utf-8 _*_
import numpy as np
from sklearn import model_selection,svm
from skfeature.function.information_theoretical_based import LCSI
from sklearn import metrics

data=np.load('leukemia_balanced_X.npy') #breast/97-24481   ovarian/253-15154  colon/62(22-40)-2000
label=np.load('leukemia_balanced_y.npy')#cns/60-7129   leukemia/72-7129  prostate/136(59-77)-12600

def mRMR(n_genes):
    f=LCSI.lcsi(data,label,function_name="MRMR",n_selected_features=n_genes,gamma=0)
    newdata=data[:,f[0]]
    return newdata

def model(n_genes):
    data=mRMR(n_genes)
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

    res_acc = [];
    res_auc = [];
    res_mcc = []
    # reduced range from 1,129 because the model was taking a very long time
    for i in range(1, 50):
        ac, uc, mc = model(i)
        res_acc.append(ac)
        res_auc.append(uc)
        res_mcc.append(mc)
    print('acc:', res_acc)
    print('auc:', res_auc)
    print('mcc:', res_mcc)  
