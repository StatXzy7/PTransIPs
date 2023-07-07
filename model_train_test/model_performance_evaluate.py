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

from BERT_model_pretrain_final import *
from bert_train_pretrain_final import BERT_encoding


from Algorithm_eval_function import LR_eval,LR_L2_eval,Linear_SVM_eval,Kernel_SVM_eval,RF_eval,DL_eval,BERT_eval,preBERT_eval
import pickle
import ml_config
import config

cf = config.get_train_config()
cf.task = 'test'
device = torch.device('cpu')
BERT_model = BERT(cf)

#-----------------ST---------------------
# train,test = data_read()
# train_seq = train.iloc[:,1]
# test_seq = test.iloc[:,1]
# # train_label = train.iloc[:,0]
# test_label = torch.tensor(np.array(test.iloc[:,0],dtype='int64')).to(device) #非常重要
# train_encoding = BERT_encoding(train_seq,test_seq)
# test_encoding = BERT_encoding(test_seq,train_seq)
# test_embedding = torch.tensor(np.load('/root/autodl-tmp/myDNAPredict/program 1.1/data/x_test_embedding.npy')).to(device)
# test_str_embedding = torch.tensor(np.load('/root/autodl-tmp/myDNAPredict/program 1.1/data/test_str_embedding.npy')).to(device)
#------------------------------------------------------------------------------------------------------------------
#----------------Y----------------------

train = pd.read_csv("/root/autodl-tmp/myDNAPredict/program 1.1/data/Y-train.csv",header=0)
test = pd.read_csv("/root/autodl-tmp/myDNAPredict/program 1.1/data/Y-test.csv",header=0)
train_seq = train.iloc[:,1]
test_seq = test.iloc[:,1]
test_label = torch.tensor(np.array(test.iloc[:,0],dtype='int64')).to(device) #非常重要
train_encoding = BERT_encoding(train_seq,test_seq)
test_encoding = BERT_encoding(test_seq,train_seq)
test_embedding = torch.tensor(np.load('/root/autodl-tmp/myDNAPredict/program 1.1/data/Y_test_embedding.npy')).to(device)
test_str_embedding = torch.tensor(np.load('/root/autodl-tmp/myDNAPredict/program 1.1/data/Y_test_str_embedding.npy')).to(device) 

#--------------------------------------

#-------------------请修改这里的文件读取路径！---------------
# path = '/root/autodl-tmp/myDNAPredict/program 1.1/dl_model_Y'
path = '/root/autodl-tmp/myDNAPredict/program 1.1/dl_model_save_nopre_Y'
# path = '/root/autodl-tmp/myDNAPredict/program 1.1/dl_model_save_noall_Y'
# path = '/root/autodl-tmp/myDNAPredict/program 1.1/dl_model_save_nostr_Y'
# path = '/root/autodl-tmp/myDNAPredict/program 1.1/dl_model_save_nostr_Y_1'
# path = '/root/autodl-tmp/myDNAPredict/program 1.1/dl_model_ST'
# path = '/root/autodl-tmp/myDNAPredict/program 1.1/dl_model_save_noall_ST'
# path = '/root/autodl-tmp/myDNAPredict/program 1.1/dl_model_save_nopre_ST'
# path = '/root/autodl-tmp/myDNAPredict/program 1.1/dl_model_save_nostr_ST'

# 获取文件夹中所有文件名
files = os.listdir(path)
# 筛选出以'.pt'结尾的文件名
pt_files = [os.path.join(path, f) for f in files if f.endswith('.pt')]
# pt_files = [os.path.join(path, "fold0_BERT_model.pt") ]
# 评估所有.pt模型
BERT_test_auc = 0
num = 0 
Result = []
Result_softmax = []
for f in pt_files:
    print('loading model ',f)
    BERT_model.load_state_dict(torch.load(f))
    BERT_model = BERT_model.to(device)
    BERT_model.eval()
    # BERT_test_acc, BERT_test_prob = preBERT_eval(BERT_model,test_encoding,test_embedding,test_label)
    
    # result, _ = BERT_model(test_encoding,test_embedding) #nostr选用这个模型
    result, _ = BERT_model(test_encoding,test_embedding, test_str_embedding) #如果使用str结构embedding选这个模型
    result_softmax = F.softmax(result, dim=1)  # Apply softmax to the output
    # print("result.shape = ",result.shape)
    # print("result_softmax = ",result_softmax.shape)
    Result.append(result)
    Result_softmax.append(result_softmax)
    
    _,predicted=torch.max(result_softmax,1)
    correct = (predicted==test_label).sum().item()
    result = result.cpu().detach().numpy()
    result_softmax = result_softmax.cpu().detach().numpy()
    BERT_test_acc, BERT_test_prob = 100*correct/result.shape[0], result_softmax

    # 计算真正例 (TP)，真负例 (TN)，假正例 (FP) 和假负例 (FN)
    tn, fp, fn, tp = confusion_matrix(test_label, predicted).ravel()
    # 计算精度 (ACC)
    BERT_test_acc = 100 * (tp + tn) / (tp + tn + fp + fn)
    # 计算敏感度 (SEN)
    BERT_sen = tp / (tp + fn)
    # 计算特异性 (SPEC)
    BERT_spec = tn / (tn + fp)
    # 计算 Matthew's Correlation Coefficient (MCC)
    BERT_mcc = matthews_corrcoef(test_label, predicted)
    # 计算 AUC
    BERT_test_auc = roc_auc_score(test_label, BERT_test_prob[:,1])
    
    result_str = 'Model file name: {}  Accuracy: {:.4f} SEN: {:.4f} SPEC: {:.4f} MCC: {:.4f} AUC: {:.4f}\n'.format(f, BERT_test_acc, BERT_sen, BERT_spec, BERT_mcc, BERT_test_auc)
    print(result_str)

    with open(os.path.join(path, 'text_result.txt'), "a") as f:
        f.write(result_str)



##
# Compute mean of Result and Result_softmax
mean_Result_softmax = np.mean([t.detach().numpy() for t in Result_softmax], axis=0)
mean_Result_softmax = torch.tensor(mean_Result_softmax)
mean_Result = np.mean([t.detach().numpy() for t in Result], axis=0)
##


# 转换每个模型的预测结果为二元标签，并进行投票
votes = [torch.argmax(t, dim=1) for t in Result_softmax]
votes = torch.stack(votes)
votes_sum = torch.sum(votes, dim=0)

predicted = torch.where(votes_sum > len(votes)/2, torch.ones_like(votes_sum), torch.zeros_like(votes_sum))

# ## 从这里需要改一下代码
# _, predicted = torch.max(mean_Result_softmax, dim=1)

correct = (predicted==test_label).sum().item()
BERT_test_acc = 100*correct/mean_Result.shape[0]
BERT_test_prob = mean_Result_softmax.detach().numpy()


# 计算预测结果
# _, predicted = torch.max(mean_Result_softmax, dim=1)

# 计算真正例 (TP)，真负例 (TN)，假正例 (FP) 和假负例 (FN)
tn, fp, fn, tp = confusion_matrix(test_label, predicted).ravel()
# 计算精度 (ACC)
BERT_test_acc = 100 * (tp + tn) / (tp + tn + fp + fn)
# 计算敏感度 (SEN)
BERT_sen = tp / (tp + fn)
# 计算特异性 (SPEC)
BERT_spec = tn / (tn + fp)
# 计算 Matthew's Correlation Coefficient (MCC)
BERT_mcc = matthews_corrcoef(test_label, predicted)
# 计算 AUC
BERT_test_auc = roc_auc_score(test_label, BERT_test_prob[:,1])

result_str = 'All kfold Model file:  Accuracy: {:.4f} SEN: {:.4f} SPEC: {:.4f} MCC: {:.4f} AUC: {:.4f}\n'.format(BERT_test_acc, BERT_sen, BERT_spec, BERT_mcc, BERT_test_auc)
print(result_str)

with open(os.path.join(path, 'text_result.txt'), "a") as f:
    f.write(result_str)

np.save(os.path.join(path, 'BERT_test_prob.npy'), BERT_test_prob)
