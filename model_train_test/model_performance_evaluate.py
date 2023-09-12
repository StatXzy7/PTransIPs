# -*- coding: utf-8 -*-
import os
import numpy as  np
import pandas as pd
import sklearn
import torch
from ml_set import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import precision_recall_curve,average_precision_score, recall_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef

from PTransIPs_model import *
from train import BERT_encoding

from Algorithm_eval_function import LR_eval,LR_L2_eval,Linear_SVM_eval,Kernel_SVM_eval,RF_eval,DL_eval,BERT_eval,preBERT_eval
import pickle
import config

cf = config.get_train_config()
cf.task = 'test'
device = torch.device('cpu')
BERT_model = BERT(cf)

# ----------------- For S/T data set ---------------------
# train,test = data_read()
# train_seq = train.iloc[:,1]
# test_seq = test.iloc[:,1]
# test_label = torch.tensor(np.array(test.iloc[:,0],dtype='int64')).to(device) # Very important
# train_encoding = BERT_encoding(train_seq,test_seq)
# test_encoding = BERT_encoding(test_seq,train_seq)
# test_embedding = torch.tensor(np.load('./data/ST_test_embedding.npy')).to(device)
# test_str_embedding = torch.tensor(np.load('./data/ST_test_str_embedding.npy')).to(device)
#-------------------Please change the directory path here to fit your model---------------
# path = './model/ST_train'
# -------------------------------------------------------

# ----------------- For Y data set ----------------------
train,test = data_readY()
train_seq = train.iloc[:,1]
test_seq = test.iloc[:,1]
test_label = torch.tensor(np.array(test.iloc[:,0],dtype='int64')).to(device) # Very important
train_encoding = BERT_encoding(train_seq,test_seq)
test_encoding = BERT_encoding(test_seq,train_seq)
test_embedding = torch.tensor(np.load('./data/Y_test_embedding.npy')).to(device)
test_str_embedding = torch.tensor(np.load('./data/Y_test_str_embedding.npy')).to(device)
#-------------------Please change the directory path here to fit your model---------------
path = './model/Y_train'
# ------------------------------------------------------

# Get all file names in the directory
files = os.listdir(path)
# Filter out file names ending with '.pt'
pt_files = [os.path.join(path, f) for f in files if f.endswith('.pt')]
# Evaluate all .pt models
BERT_test_auc = 0
num = 0 
Result = []
Result_softmax = []
for f in pt_files:
    print('loading model ',f)
    BERT_model.load_state_dict(torch.load(f))
    BERT_model = BERT_model.to(device)
    BERT_model.eval()
    
    # Use the following line for models without the structure embedding
    # result, _ = BERT_model(test_encoding,test_embedding)
    # Use the following line for models with the structure embedding
    result, _ = BERT_model(test_encoding,test_embedding, test_str_embedding)
    result_softmax = F.softmax(result, dim=1)  # Apply softmax to the output
    
    Result.append(result)
    Result_softmax.append(result_softmax)
    
    _,predicted=torch.max(result_softmax,1)
    correct = (predicted==test_label).sum().item()
    result = result.cpu().detach().numpy()
    result_softmax = result_softmax.cpu().detach().numpy()
    BERT_test_acc, BERT_test_prob = 100*correct/result.shape[0], result_softmax

    # Calculate True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN)
    tn, fp, fn, tp = confusion_matrix(test_label, predicted).ravel()
    # Calculate accuracy (ACC)
    BERT_test_acc = 100 * (tp + tn) / (tp + tn + fp + fn)
    # Calculate sensitivity (SEN)
    BERT_sen = tp / (tp + fn)
    # Calculate specificity (SPEC)
    BERT_spec = tn / (tn + fp)
    # Calculate Matthew's Correlation Coefficient (MCC)
    BERT_mcc = matthews_corrcoef(test_label, predicted)
    # Calculate AUC
    BERT_test_auc = roc_auc_score(test_label, BERT_test_prob[:,1])
    
    result_str = 'Model file name: {}  Accuracy: {:.4f} SEN: {:.4f} SPEC: {:.4f} MCC: {:.4f} AUC: {:.4f}\n'.format(f, BERT_test_acc, BERT_sen, BERT_spec, BERT_mcc, BERT_test_auc)
    print(result_str)

    with open(os.path.join(path, 'PTransIPs_text_result.txt'), "a") as f:
        f.write(result_str)

# Compute mean of Result and Result_softmax
mean_Result_softmax = np.mean([t.detach().numpy() for t in Result_softmax], axis=0)
mean_Result_softmax = torch.tensor(mean_Result_softmax)
mean_Result = np.mean([t.detach().numpy() for t in Result], axis=0)

# Convert predictions from each model to binary labels and vote
votes = [torch.argmax(t, dim=1) for t in Result_softmax]
votes = torch.stack(votes)
votes_sum = torch.sum(votes, dim=0)

predicted = torch.where(votes_sum > len(votes)/2, torch.ones_like(votes_sum), torch.zeros_like(votes_sum))

# Calculate True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN)
tn, fp, fn, tp = confusion_matrix(test_label, predicted).ravel()
# Calculate accuracy (ACC)
BERT_test_acc = 100 * (tp + tn) / (tp + tn + fp + fn)
# Calculate sensitivity (SEN)
BERT_sen = tp / (tp + fn)
# Calculate specificity (SPEC)
BERT_spec = tn / (tn + fp)
# Calculate Matthew's Correlation Coefficient (MCC)
BERT_mcc = matthews_corrcoef(test_label, predicted)
# Calculate AUC
BERT_test_auc = roc_auc_score(test_label, BERT_test_prob[:,1])

result_str = 'All kfold Model file:  Accuracy: {:.4f} SEN: {:.4f} SPEC: {:.4f} MCC: {:.4f} AUC: {:.4f}\n'.format(BERT_test_acc, BERT_sen, BERT_spec, BERT_mcc, BERT_test_auc)
print(result_str)

with open(os.path.join(path, 'PTransIPs_text_result.txt'), "a") as f:
    f.write(result_str)

np.save(os.path.join(path, 'PTransIPs_test_prob.npy'), BERT_test_prob)