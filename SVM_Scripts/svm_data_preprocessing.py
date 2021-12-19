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

def discret():
    data = np.load('leukemia_balanced_X.npy')
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
    np.save('discret_leukemia_breast.npy',data)

###########  Balance the raw data with RVOS method

def data_generator_arff():
    # get the raw data set and separate it based on class label
    raw = arff.load(open('leukemia.arff', 'rb'))
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
    np.save('leukemia_balanced_X',leukemia_X)
    np.save('leukemia_balanced_y', leukemia_y)


if __name__=='__main__':
     # readarff()
     # discret()
    # readMat()
    # readCsv()
    data_generator_arff()
    # data_generateor_mat()
    #  data_generateor_csv()

