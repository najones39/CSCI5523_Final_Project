# _*_ coding=utf-8 _*_
import numpy as np
from sklearn import model_selection,svm,linear_model
from matplotlib import pyplot as plt

data=np.load('Leukemia_balanced_X.npy')
label=np.load('Leukemia_balanced_y.npy')

def my_svm_rfe(n_selected_features):
    X=data
    y=label
    m, n_features = X.shape
    clf = svm.LinearSVC(penalty='l1',loss='squared_hinge',C=0.1,dual=False,
                        fit_intercept=True,random_state=1)
    count = n_features
    cut=n_features
    n_steps=600
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
def getgene(n):
    data = my_svm_rfe(n)
    fea = [19]
    lr = linear_model.LogisticRegression(n_jobs=3, penalty='l2', solver='liblinear', C=19)
    train_scores, test_scores = model_selection.validation_curve(
        lr, data, label, param_name="C", param_range=fea,
        cv=5, scoring="accuracy", n_jobs=3)
    train_scores_mean = np.mean(train_scores, axis=1).tolist()
    test_scores_mean = np.mean(test_scores, axis=1).tolist()
    return train_scores_mean, test_scores_mean
def getgenes():
    train=[];test=[]
    for i in range(8):
        tra,tes=getgene(2**i)
        train.extend(tra)
        test.extend(tes)
    return train,test
def validationCurves():
    genes=range(8)
    train_scores_mean ,test_scores_mean = getgenes()

    plt.title("Validation Curve on Leukemia", fontsize=16)
    plt.xlabel("Number of genes selected", fontsize=16)
    plt.ylabel("Cross validation scores", fontsize=16)
    plt.ylim(0.9, 1.005)
    lw = 2

    plt.plot(genes,train_scores_mean,'r-o',label="Training scores", lw=lw)
    plt.plot(genes,test_scores_mean, 'b-o',label="Testing scores",  lw=lw)
    plt.xticks(genes,('1','2','4','8','16','32','64','128'))
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig('Leukemia.pdf')
    plt.show()

if __name__ == '__main__':

    validationCurves()