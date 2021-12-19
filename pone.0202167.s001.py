#  _*_ coding=utf-8 _*_
import numpy as np
from scipy import io
import random
import arff
import csv
from sklearn import preprocessing

def readarff():
    raw=arff.load(open('leukemia.arff','rb'))
    data=raw['data']
    for row in data:
        if row[-1] == 'Normal':
            row[-1] = 1
        else:row[-1] = 0
    # random.shuffle(data)
    data=np.array(data)
    data_data = data[:, :-1];data_label = data[:, -1]
    data_data=preprocessing.scale(data_data)
    data_label=data_label-np.mean(data_label,axis=0)
    np.save('leukemia_data',data_data)
    np.save('leukemia_label',data_label)

def readMat():
    file=io.loadmat('colon.mat')
    data=file['X']
    label=file['Y']
    ls=[]
    for row in label:
        ls.append(row[0])
    data=np.array(data)
    data=preprocessing.scale(data)
    label=np.array(ls)
    np.save('colon_data',data)
    np.save('colon_label',label)
    print(data,label)
def readCsv():
    ls=[];dataset=[]
    fte=file('prostate_TumorVSNormal_test.csv','r')
    test=csv.reader(fte)
    for row in test:
        clist=[char for char in row]
        ls.append(clist)
    for r in ls[1:]:
        dataset.append([float(num) for num in r])
    fte.close()

    l = [];
    ftr = file('prostate_TumorVSNormal_train.csv', 'r')
    train = csv.reader(ftr)
    for row in train:
        clist = [char for char in row]
        l.append(clist)
    for r in l[1:]:
        dataset.append([float(num) for num in r])
    ftr.close()

    dataset=np.array(dataset)
    # random.shuffle(dataset)
    data=preprocessing.scale(dataset[:, 1:-1])
    label=dataset[:, 0]
    np.save('prostate_data',data)
    np.save('prostate_label',label)

def discret():
    data = np.load('breast_balanced_X.npy')
    # label = np.load('leukemia_balanced_y.npy')
    row,col=data.shape
    for i in xrange(col):
        m=np.mean(data[:,i])
        u=np.std(data[:,i])
        for j in xrange(row):
            if data[j][i]>(m+u/2):
                data[j][i]=2
            elif data[j][i]<(m-u/2):
                data[j][i]=-2
            else:data[j][i]=0
    np.save('discret_balanced_breast.npy',data)

###########  Balance the raw data with RVOS method

def data_generator_arff():
    # get the raw data set and separate it based on class label
    raw = arff.load(open('ovarian.arff', 'rb'))
    data = raw['data']
    for row in data:
        if row[-1] == 'Normal':
            row[-1] = 1
        else:
            row[-1] = 0
    leukemia_1=[];leukemia_0=[]
    for sample in data:
        if sample[-1]==1:
            leukemia_1.append(sample)
        else:leukemia_0.append(sample)
    leukemia_1=np.array(leukemia_1)
    leukemia_0=np.array(leukemia_0)

    # generate sub_dataset and extend it to the small one
    m_1,n_1=leukemia_1.shape
    m_0,n_0=leukemia_0.shape
    leukemia_1 = leukemia_1.tolist()
    leukemia_0=leukemia_0.tolist()
    dataset_generated=[]
    count=m_0-m_1
    while count:
        sample = []
        for j in xrange(n_1):
            element=leukemia_1[random.randint(0,m_1-1)][j]
            sample.append(element)
        dataset_generated.append(sample)
        count-=1
    leukemia_1.extend(dataset_generated)
    # merge two data sets coming from different classes
    leukemia_1.extend(leukemia_0)
    random.shuffle(leukemia_1)
    leukemia_1 = np.array(leukemia_1)
    # preprocess and save
    leukemia_X=leukemia_1[:,:-1]
    leukemia_y=leukemia_1[:,-1]
    leukemia_X=preprocessing.scale(leukemia_X)
    np.save('ovarian_balanced_X',leukemia_X)
    np.save('ovarian_balanced_y', leukemia_y)

def data_generateor_mat():
    # get the raw data set and separate it based on class label
    file = io.loadmat('colon.mat')
    data = file['X']
    label = file['Y']
    data = np.array(data)
    data=data.tolist()
    ls = [];
    for row in label:
        ls.append(row[0])
    for line in data:
        line.append(ls[data.index(line)])
    colon_1=[];colon_0=[]
    for sample in data:
        if sample[-1]==1:
            colon_1.append(sample)
        else:colon_0.append(sample)
    colon_0=np.array(colon_0)
    colon_1=np.array(colon_1)

    # generate sub_dataset and extend it to the small one
    m_1,n_1=colon_1.shape;
    m_0,n_0=colon_0.shape;
    colon_1 = colon_1.tolist()
    colon_0=colon_0.tolist()
    dataset_generated=[]
    count=m_0-m_1
    while count:
        sample = []
        for j in xrange(n_1):
            element=colon_1[random.randint(0,m_1-1)][j]
            sample.append(element)
        dataset_generated.append(sample)
        count-=1
    colon_1.extend(dataset_generated)
    # merge two data sets coming from different classes
    colon_1.extend(colon_0)
    random.shuffle(colon_1)
    colon_1 = np.array(colon_1)
    # preprocess and save
    colon_X=colon_1[:,:-1]
    colon_y=colon_1[:,-1]
    colon_X=preprocessing.scale(colon_X)
    np.save('colon_balanced_X',colon_X)
    np.save('colon_balanced_y', colon_y)

def data_generateor_csv():
    # get the raw data set and separate it based on class label
    ls = [];
    dataset = []
    fte = file('prostate_TumorVSNormal_test.csv', 'r')
    test = csv.reader(fte)
    for row in test:
        clist = [char for char in row]
        ls.append(clist)
    for r in ls[1:]:
        dataset.append([float(num) for num in r])
    fte.close()

    l = [];
    ftr = file('prostate_TumorVSNormal_train.csv', 'r')
    train = csv.reader(ftr)
    for row in train:
        clist = [char for char in row]
        l.append(clist)
    for r in l[1:]:
        dataset.append([float(num) for num in r])
    ftr.close()
    dataset = np.array(dataset)
    dataset.tolist()
    cns_1 = [];
    cns_0 = []
    for sample in dataset:
        if sample[0] == 2:
            cns_1.append(sample)
        else:
            cns_0.append(sample)
    cns_1 = np.array(cns_1)
    cns_0 = np.array(cns_0)

    # generate sub_dataset and extend it to the small one
    m_1,n_1=cns_1.shape;
    m_0,n_0=cns_0.shape;
    cns_1 = cns_1.tolist()
    cns_0=cns_0.tolist()
    dataset_generated=[]
    count=m_0-m_1
    while count:
        sample = []
        for j in xrange(n_1):
            element=cns_1[random.randint(0,m_1-1)][j]
            sample.append(element)
        dataset_generated.append(sample)
        count-=1
    cns_1.extend(dataset_generated)
    # merge two data sets coming from different classes
    cns_1.extend(cns_0)
    random.shuffle(cns_1)
    cns_1 = np.array(cns_1)
    # preprocess and save
    cns_X=cns_1[:,1:]
    cns_y=cns_1[:,0]
    cns_X=preprocessing.scale(cns_X)
    np.save('prostate_balanced_X',cns_X)
    np.save('prostate_balanced_y', cns_y)

if __name__=='__main__':
     # readarff()
     # discret()
    # readMat()
    # readCsv()
    data_generator_arff()
    # data_generateor_mat()
    #  data_generateor_csv()

