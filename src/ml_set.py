# -*- coding: utf-8 -*-
"""
@author: ZiyangXu
"""

import numpy as  np
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import svm


def data_read():
    train = pd.read_csv('./data/ST-train.csv',header=0)
    test = pd.read_csv('./data/ST-test.csv',header=0)
    return train,test

def data_readY():
    train = pd.read_csv('./data/Y-train.csv',header=0)
    test = pd.read_csv('./data/Y-test.csv',header=0)
    return train,test

def embedding_load():
    x_train_embedding = torch.tensor(np.load('./embedding/ST_train_embedding.npy')).to('cuda')
    x_test_embedding = torch.tensor(np.load('./embedding/ST_test_embedding.npy')).to('cuda')
    return  x_train_embedding,x_test_embedding
def embedding_str_load():
    x_train_str_embedding = torch.tensor(np.load('./embedding/ST_train_str_embedding.npy')).to('cuda')
    x_test_str_embedding = torch.tensor(np.load('./embedding/ST_test_str_embedding.npy')).to('cuda') 
    return  x_train_str_embedding,x_test_str_embedding

def embedding_loadY():
    x_train_embedding = torch.tensor(np.load('./embedding/Y_train_embedding.npy')).to('cuda')
    x_test_embedding = torch.tensor(np.load('./embedding/Y_test_embedding.npy')).to('cuda')
    return  x_train_embedding,x_test_embedding

def embedding_str_loadY():
    x_train_str_embedding = torch.tensor(np.load('./embedding/Y_train_str_embedding.npy')).to('cuda')
    x_test_str_embedding = torch.tensor(np.load('./embedding/Y_test_str_embedding.npy')).to('cuda') 
    return  x_train_str_embedding,x_test_str_embedding



def encoding(txt_array,test_array):
    txt_seq_length = 0
    test_seq_length = 0
    txt_number =len(txt_array)
    test_number = len(test_array)
    for i in range(txt_number):
        if len(txt_array[i]) > txt_seq_length:
            txt_seq_length = len(txt_array[i])
    for i in range(test_number):
        if len(test_array[i]) > test_seq_length:
            test_seq_length = len(test_array[i])
    seq_length = max(txt_seq_length,test_seq_length)
    x = np.zeros([txt_number, seq_length])
    nuc_d = {'0':0,'A':1,'B':22,'U':23,'J':24,'Z':25,'O':26,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,'X':21}
    for seq_index in range(txt_number):
        seq = txt_array[seq_index].upper()
        seq = seq[ 0 - seq_length:].ljust(seq_length,'0')
    
        for n, base in enumerate(seq):
            x[seq_index][n] = nuc_d[base]
    x = np.array(x,dtype='int64')
    x = torch.LongTensor(x)
    x  = F.one_hot(x)
    x = torch.flatten(x,1)
    x = np.array(x,dtype='float32')
    return x

def test_acc(predict_label,true_label):
    correct = 0
    correct += (predict_label==true_label).sum().item()
    return 100*correct/len(predict_label)



#LR_def

LR = LogisticRegression(penalty='none',fit_intercept=True,max_iter=500,tol =10e-4 )
#LR_L1 = LogisticRegression(penalty='l1',fit_intercept=True,max_iter=500,tol =10e-4 )
LR_L2 = LogisticRegression(penalty='l2',fit_intercept=True,max_iter=500,tol =10e-4 )
RF = RF(n_estimators=400,max_depth=10,criterion='gini')
Linear_SVM = svm.SVC(kernel='linear',gamma='auto',tol=0.001)
Kernel_SVM = svm.SVC(kernel='rbf',gamma='auto',tol=0.001)


