# -*- coding: utf-8 -*-
"""
@author: ZiyangXu
"""
import numpy as np
import pandas as pd
import torch
import sklearn
import torch.nn.functional as F
from ml_set import *
from simple_dl_set import *

def LR_eval(train_encoding,train_label,test_encoding,test_label):
    LR_train = LR.fit(train_encoding,train_label)
    LR_predict = LR.predict(test_encoding)
    LR_test_acc = test_acc(LR_predict,test_label)
    LR_test_prob = LR.predict_proba(test_encoding)
    return  LR_test_acc,LR_test_prob



# def LR_L1_eval(train_encoding,train_label,test_encoding,test_label):
#     LR_train = LR_L1.fit(train_encoding,train_label)
#     LR_predict = LR_L1.predict(test_encoding)
#     LR_test_acc = test_acc(LR_predict,test_label)
#     LR_test_prob = LR_L1.predict_proba(test_encoding)
#     return  LR_test_acc,LR_test_prob



def LR_L2_eval(train_encoding,train_label,test_encoding,test_label):
    LR_train = LR_L2.fit(train_encoding,train_label)
    LR_predict = LR_L2.predict(test_encoding)
    LR_test_acc = test_acc(LR_predict,test_label)
    LR_test_prob = LR_L2.predict_proba(test_encoding)
    return  LR_test_acc,LR_test_prob


def Linear_SVM_eval(train_encoding,train_label,test_encoding,test_label):
    SVM_train = Linear_SVM.fit(train_encoding,train_label)
    SVM_predict = Linear_SVM.predict(test_encoding)
    SVM_test_acc = test_acc(SVM_predict,test_label)
    SVM_test_prob = Linear_SVM.decision_function(test_encoding)
    return  SVM_test_acc,SVM_test_prob

def Kernel_SVM_eval(train_encoding,train_label,test_encoding,test_label):
    SVM_train = Kernel_SVM.fit(train_encoding,train_label)
    SVM_predict = Kernel_SVM.predict(test_encoding)
    SVM_test_acc = test_acc(SVM_predict,test_label)
    SVM_test_prob = Kernel_SVM.decision_function(test_encoding)
    return  SVM_test_acc,SVM_test_prob


def RF_eval(train_encoding,train_label,test_encoding,test_label):
    RF_train = RF.fit(train_encoding,train_label)
    RF_predict = RF.predict(test_encoding)
    RF_test_acc = test_acc(RF_predict,test_label)
    RF_test_prob = RF.predict_proba(test_encoding)
    return  RF_test_acc,RF_test_prob


def DL_eval(model,test,test_labels):
    
    Result = model(test)
    _,predicted=torch.max(Result,1)
    correct = 0
    correct += (predicted==test_labels).sum().item()
    Result = Result.cpu().detach().numpy()
    #Result = np.array(Result)
    return 100*correct/Result.shape[0],Result


def BERT_eval(model,test,test_labels):
    Result,out = model(test)
    Result = F.softmax(Result)
    _,predicted=torch.max(Result,1)
    correct = 0
    correct += (predicted==test_labels).sum().item()
    Result = Result.cpu().detach().numpy()
    #Result = np.array(Result)
    return 100*correct/Result.shape[0],Result

def preBERT_eval(model, test, test_embedding, test_labels):
    Result, _ = model(test,test_embedding)
    Result_softmax = F.softmax(Result, dim=1)  # Apply softmax to the output
    _,predicted=torch.max(Result_softmax,1)
    correct = (predicted==test_labels).sum().item()
    Result = Result.cpu().detach().numpy()
    Result_softmax = Result_softmax.cpu().detach().numpy()
    #Result = np.array(Result)
    return 100*correct/Result.shape[0], Result_softmax
    # return 100*correct/Result.shape[0], predicted
    
def preBERT_eval(model, test, test_embedding, test_labels):
    Result, _ = model(test,test_embedding)
    Result_softmax = F.softmax(Result, dim=1)  # Apply softmax to the output
    _,predicted=torch.max(Result_softmax,1)
    correct = (predicted==test_labels).sum().item()
    Result = Result.cpu().detach().numpy()
    Result_softmax = Result_softmax.cpu().detach().numpy()
    #Result = np.array(Result)
    return 100*correct/Result.shape[0], Result_softmax