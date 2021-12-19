# _*_ coding=utf-8 _*_
import numpy as np
from sklearn import model_selection,svm,linear_model,naive_bayes
from skfeature.function.similarity_based import reliefF
from sklearn import metrics

raw_data=np.load('leukemia_data.npy')
label=np.load('leukemia_label.npy')

label = label>0

def inf_reliefF(n_genes):
    score=reliefF.reliefF(raw_data,label)
    index = score #(reliefF.feature_ranking(score))[:n_genes]
    ls = list(range(len(score)))  #(range(0, len(score))
    for i in range(len(score)):
        if i not in index:
            ls[i] = False
        else:
            ls[i] = True
    newdata = raw_data[:, np.array(ls)]
    return newdata

def model(n_genes):
    data=inf_reliefF(n_genes)
    ls_acc=[];ls_mcc=[];ls_auc=[]
    skf=model_selection.StratifiedKFold(n_splits=5,shuffle=False,random_state=None)
    fs = svm.SVC(kernel='linear',C=1.0,probability=True)
    for train_index, test_index in skf.split(data,label):
        fs.fit(data[train_index], label[train_index])
        acc=fs.score(data[test_index],label[test_index])
        ls_acc.append(acc)
        mcc=metrics.matthews_corrcoef(label[test_index],fs.predict(data[test_index]))
        ls_mcc.append(mcc)
        fpr, tpr, thresholds = metrics.roc_curve(label[test_index], (fs.predict_proba(data[test_index]))[:,1])
        auc=metrics.auc(fpr,tpr)
        ls_auc.append(auc)
    return np.mean(ls_acc),np.mean(ls_auc),np.mean(ls_mcc)

if __name__ == '__main__':
    res_acc = [];
    res_auc = [];
    res_mcc = []
    for i in range(1, 129):  #129
        ac, uc, mc = model(i)
        res_acc.append(ac)
        res_auc.append(uc)
        res_mcc.append(mc)
    print('acc:', res_acc)
    print('auc:', res_auc)
    print('mcc:', res_mcc)

