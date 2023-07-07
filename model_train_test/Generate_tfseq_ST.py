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
# from simple_dl_set_DeepIPS import *
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
train,test = data_read()

train_seq = train.iloc[:,1]
test_seq = test.iloc[:,1]
train_label = torch.tensor(np.array(train.iloc[:,0],dtype='int64')).to(device) #非常重要
test_label = torch.tensor(np.array(test.iloc[:,0],dtype='int64')).to(device) #非常重要
train_encoding = BERT_encoding(train_seq,test_seq)
test_encoding = BERT_encoding(test_seq,train_seq)
train_embedding = torch.tensor(np.load('/root/autodl-tmp/myDNAPredict/program 1.1/data/x_train_embedding.npy')).to(device)
test_embedding = torch.tensor(np.load('/root/autodl-tmp/myDNAPredict/program 1.1/data/x_test_embedding.npy')).to(device)
train_str_embedding = torch.tensor(np.load('/root/autodl-tmp/myDNAPredict/program 1.1/data/train_str_embedding.npy')).to(device)
test_str_embedding = torch.tensor(np.load('/root/autodl-tmp/myDNAPredict/program 1.1/data/test_str_embedding.npy')).to(device)

# Concatenate train and test sequences
test_seq = pd.concat([train_seq, test_seq])
# Concatenate train and test labels
test_label = torch.cat([train_label, test_label])
# Concatenate train and test encodings
test_encoding = torch.cat([train_encoding, test_encoding])
# Concatenate train and test embeddings
test_embedding = torch.cat([train_embedding, test_embedding])
# Concatenate train and test str embeddings
test_str_embedding = torch.cat([train_str_embedding, test_str_embedding])


#------------------------------------------------------------------------------------------------------------------
#----------------Y----------------------

# train = pd.read_csv("/root/autodl-tmp/myDNAPredict/program 1.1/data/Y-train.csv",header=0)
# test = pd.read_csv("/root/autodl-tmp/myDNAPredict/program 1.1/data/Y-test.csv",header=0)
# train_seq = train.iloc[:,1]
# test_seq = test.iloc[:,1]
# test_label = torch.tensor(np.array(test.iloc[:,0],dtype='int64')).to(device) #非常重要
# train_encoding = BERT_encoding(train_seq,test_seq)
# test_encoding = BERT_encoding(test_seq,train_seq)
# test_embedding = torch.tensor(np.load('/root/autodl-tmp/myDNAPredict/program 1.1/data/Y_test_embedding.npy')).to(device)
# test_str_embedding = torch.tensor(np.load('/root/autodl-tmp/myDNAPredict/program 1.1/data/Y_test_str_embedding.npy')).to(device) 

#--------------------------------------

#-------------------请修改这里的文件读取路径！---------------
path = '/root/autodl-tmp/myDNAPredict/program 1.1/dl_model_ST'

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

# for i, f in enumerate(pt_files):
#     print('loading model ',f)
#     BERT_model.load_state_dict(torch.load(f))
#     BERT_model = BERT_model.to(device)
#     BERT_model.eval()

#     result, _ = BERT_model(test_encoding,test_embedding, test_str_embedding)
#     result_softmax = F.softmax(result, dim=1)

#     Result.append(result)
#     Result_softmax.append(result_softmax)
    
#     _,predicted=torch.max(result_softmax,1)
#     correct = (predicted==test_label).sum().item()
#     result = result.cpu().detach().numpy()
#     result_softmax = result_softmax.cpu().detach().numpy()

#     # Save result
#     np.save(os.path.join(path, f"result_{i}"), result)

#     BERT_test_acc, BERT_test_prob = 100*correct/result.shape[0], result_softmax
#     tn, fp, fn, tp = confusion_matrix(test_label, predicted).ravel()
#     BERT_test_acc = 100 * (tp + tn) / (tp + tn + fp + fn)
#     BERT_sen = tp / (tp + fn)
#     BERT_spec = tn / (tn + fp)
#     BERT_mcc = matthews_corrcoef(test_label, predicted)
#     BERT_test_auc = roc_auc_score(test_label, BERT_test_prob[:,1])
    
#     result_str = 'Model file name: {}  Accuracy: {:.4f} SEN: {:.4f} SPEC: {:.4f} MCC: {:.4f} AUC: {:.4f}\n'.format(f, BERT_test_acc, BERT_sen, BERT_spec, BERT_mcc, BERT_test_auc)
#     print(result_str)

# 筛选出以'result'开头且以'.npy'结尾的文件名
result_files = [f for f in files if f.startswith('result') and f.endswith('.npy')]
# 对筛选出的文件进行排序，保证它们的顺序与之前保存时的顺序一致
result_files.sort()
# 创建一个空列表用于存储结果
Result = []
# 遍历所有的结果文件
for f in result_files:
    # 加载.npy文件
    result = np.load(os.path.join(path, f))
    result = torch.from_numpy(result)
    result_softmax = F.softmax(result, dim=1)
    # 添加到结果列表
    Result.append(result)
    Result_softmax.append(result_softmax)

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

correct = (predicted==test_label).sum().item()
BERT_test_acc = 100*correct/mean_Result.shape[0]
BERT_test_prob = mean_Result_softmax.detach().numpy()

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


# 通过比较 test_label 和 predicted 来得到具体的阳性和阴性预测正确和错误的具体序列
correct_predictions = (predicted == test_label)
incorrect_predictions = (predicted != test_label)

# Compute indices of correct and incorrect predictions
correct_indices = (predicted == test_label).nonzero(as_tuple=True)
incorrect_indices = (predicted != test_label).nonzero(as_tuple=True)

# Compute indices of true positive (TP), true negative (TN), false positive (FP), and false negative (FN)
TP_indices = (predicted & correct_predictions).nonzero(as_tuple=True)
TN_indices = (~predicted & correct_predictions).nonzero(as_tuple=True)
FP_indices = (predicted & incorrect_predictions).nonzero(as_tuple=True)
FN_indices = (~predicted & incorrect_predictions).nonzero(as_tuple=True)

# Get the sequences corresponding to correct and incorrect predictions
TP_sequences = test_seq.iloc[TP_indices]
TN_sequences = test_seq.iloc[TN_indices]
FP_sequences = test_seq.iloc[FP_indices]
FN_sequences = test_seq.iloc[FN_indices]

# print("TP sequences: ", TP_sequences.head(10))
# print("TN sequences: ", TN_sequences.head(10))
# print("FP sequences: ", FP_sequences.head(10))
# print("FN sequences: ", FN_sequences.head(10))

print("Number of TP sequences: ", len(TP_sequences))
print("Number of TN sequences: ", len(TN_sequences))
print("Number of FP sequences: ", len(FP_sequences))
print("Number of FN sequences: ", len(FN_sequences))

# 定义一个函数，删除序列末尾的"2"
def remove_last_character(sequences):
    return sequences.str.slice(0, -1)

# 处理四个序列，删除每个序列最后的"2"
TP_sequences = remove_last_character(TP_sequences)
TN_sequences = remove_last_character(TN_sequences)
FP_sequences = remove_last_character(FP_sequences)
FN_sequences = remove_last_character(FN_sequences)

print("TP sequences: ", TP_sequences.head(10))
print("TN sequences: ", TN_sequences.head(10))
print("FP sequences: ", FP_sequences.head(10))
print("FN sequences: ", FN_sequences.head(10))

# 定义一个函数，将序列保存为txt文件
def save_to_txt(sequences, filename):
    sequences.to_csv(filename, index=False, header=False)

# 保存序列到指定的路径
save_to_txt(TP_sequences, "myDNAPredict/program 1.1/sequence_to_analysis/ST_TP.txt")
save_to_txt(TN_sequences, "myDNAPredict/program 1.1/sequence_to_analysis/ST_TN.txt")
save_to_txt(FP_sequences, "myDNAPredict/program 1.1/sequence_to_analysis/ST_FP.txt")
save_to_txt(FN_sequences, "myDNAPredict/program 1.1/sequence_to_analysis/ST_FN.txt")

