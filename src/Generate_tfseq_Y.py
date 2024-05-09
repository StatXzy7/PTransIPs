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
import config

cf = config.get_train_config()
cf.task = 'test'
device = torch.device('cpu')
BERT_model = BERT(cf)

# #-----------------ST---------------------
# # Read training and test data
# train, test = data_read()
# # Extract sequences and labels from training and test data
# train_seq = train.iloc[:, 1]
# test_seq = test.iloc[:, 1]
# train_label = torch.tensor(np.array(train.iloc[:, 0], dtype='int64')).to(device)  # Very important
# test_label = torch.tensor(np.array(test.iloc[:, 0], dtype='int64')).to(device)  # Very important
# # Encode sequences using BERT
# train_encoding = BERT_encoding(train_seq, test_seq)
# test_encoding = BERT_encoding(test_seq, train_seq)
# # Load pre-trained embeddings
# train_embedding = torch.tensor(np.load('./data/x_train_embedding.npy')).to(device)
# test_embedding = torch.tensor(np.load('./data/x_test_embedding.npy')).to(device)
# train_str_embedding = torch.tensor(np.load('./data/train_str_embedding.npy')).to(device)
# test_str_embedding = torch.tensor(np.load('./data/test_str_embedding.npy')).to(device)

#------------------------------------------------------------------------------------------------------------------
#----------------Y----------------------
train,test = data_readY()
train = pd.read_csv("./data/Y-train.csv",header=0)
test = pd.read_csv("./data/Y-test.csv",header=0)
train_seq = train.iloc[:,1]
test_seq = test.iloc[:,1]
train_label = torch.tensor(np.array(train.iloc[:,0],dtype='int64')).to(device) #非常重要
test_label = torch.tensor(np.array(test.iloc[:,0],dtype='int64')).to(device) #非常重要
train_encoding = BERT_encoding(train_seq,test_seq)
test_encoding = BERT_encoding(test_seq,train_seq)
train_embedding = torch.tensor(np.load('./data/Y_train_embedding.npy')).to(device)
test_embedding = torch.tensor(np.load('./data/Y_test_embedding.npy')).to(device)
train_str_embedding = torch.tensor(np.load('./data/Y_train_str_embedding.npy')).to(device)
test_str_embedding = torch.tensor(np.load('./data/Y_test_str_embedding.npy')).to(device) 

#--------------------------------------

# Concatenate train and test sequences and labels
test_seq = pd.concat([train_seq, test_seq])
test_label = torch.cat([train_label, test_label])
test_encoding = torch.cat([train_encoding, test_encoding])
test_embedding = torch.cat([train_embedding, test_embedding])
test_str_embedding = torch.cat([train_str_embedding, test_str_embedding])

#-------------------Please change the file path here to fit your model---------------
path = './model/Y'

# Get all filenames in the folder
files = os.listdir(path)
# Filter out filenames that end with '.pt'
pt_files = [os.path.join(path, f) for f in files if f.endswith('.pt')]
# pt_files = [os.path.join(path, "fold0_BERT_model.pt") ]
# Evaluate all the .pt models
BERT_test_auc = 0
num = 0 
Result = []
Result_softmax = []

for i, f in enumerate(pt_files):
    print('loading model ',f)
    BERT_model.load_state_dict(torch.load(f))
    BERT_model = BERT_model.to(device)
    BERT_model.eval()

    result, _ = BERT_model(test_encoding,test_embedding, test_str_embedding)
    result_softmax = F.softmax(result, dim=1)

    # Result.append(result)
    # Result_softmax.append(result_softmax)
    
    _,predicted=torch.max(result_softmax,1)
    correct = (predicted==test_label).sum().item()
    result = result.cpu().detach().numpy()
    result_softmax = result_softmax.cpu().detach().numpy()

    # Save result
    np.save(os.path.join(path, f"result_{i}"), result)

    BERT_test_acc, BERT_test_prob = 100*correct/result.shape[0], result_softmax
    tn, fp, fn, tp = confusion_matrix(test_label, predicted).ravel()
    BERT_test_acc = 100 * (tp + tn) / (tp + tn + fp + fn)
    BERT_sen = tp / (tp + fn)
    BERT_spec = tn / (tn + fp)
    BERT_mcc = matthews_corrcoef(test_label, predicted)
    BERT_test_auc = roc_auc_score(test_label, BERT_test_prob[:,1])
    
    result_str = 'Model file name: {}  Accuracy: {:.4f} SEN: {:.4f} SPEC: {:.4f} MCC: {:.4f} AUC: {:.4f}\n'.format(f, BERT_test_acc, BERT_sen, BERT_spec, BERT_mcc, BERT_test_auc)
    print(result_str)

# Filter out filenames that start with 'result' and end with '.npy'
result_files = [f for f in files if f.startswith('result') and f.endswith('.npy')]
# Sort the filtered files to ensure they are in the same order as when they were saved
result_files.sort()
# Initialize an empty list to store the results
Result = []
# Loop through all result files
for f in result_files:
    # Load .npy files
    result = np.load(os.path.join(path, f))
    result = torch.from_numpy(result)
    result_softmax = F.softmax(result, dim=1)
    # Add to result
    Result.append(result)
    Result_softmax.append(result_softmax)

##
# Compute mean of Result and Result_softmax
mean_Result_softmax = np.mean([t.detach().numpy() for t in Result_softmax], axis=0)
mean_Result_softmax = torch.tensor(mean_Result_softmax)
mean_Result = np.mean([t.detach().numpy() for t in Result], axis=0)
##


# Convert each model's prediction to binary labels and perform voting
votes = [torch.argmax(t, dim=1) for t in Result_softmax]
votes = torch.stack(votes)
votes_sum = torch.sum(votes, dim=0)

predicted = torch.where(votes_sum > len(votes)/2, torch.ones_like(votes_sum), torch.zeros_like(votes_sum))

correct = (predicted==test_label).sum().item()
BERT_test_acc = 100*correct/mean_Result.shape[0]
BERT_test_prob = mean_Result_softmax.detach().numpy()

# Calculate True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)
tn, fp, fn, tp = confusion_matrix(test_label, predicted).ravel()
# Calculate Accuracy (ACC)
BERT_test_acc = 100 * (tp + tn) / (tp + tn + fp + fn)
# Calculate Sensitivity (SEN)
BERT_sen = tp / (tp + fn)
# Calculate Specificity (SPEC)
BERT_spec = tn / (tn + fp)
# Calculate Matthew's Correlation Coefficient (MCC)
BERT_mcc = matthews_corrcoef(test_label, predicted)
# Calculate AUC (Area Under Curve)
BERT_test_auc = roc_auc_score(test_label, BERT_test_prob[:,1])

result_str = 'All kfold Model file:  Accuracy: {:.4f} SEN: {:.4f} SPEC: {:.4f} MCC: {:.4f} AUC: {:.4f}\n'.format(BERT_test_acc, BERT_sen, BERT_spec, BERT_mcc, BERT_test_auc)
print(result_str)


# Compare test_label and predicted to get the specific sequences that are correctly and incorrectly predicted as positives and negatives
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

# Define a function to remove the last character "2" from sequences
def remove_last_character(sequences):
    return sequences.str.slice(0, -1)

# Process the four types of sequences by removing the last character "2" from each
TP_sequences = remove_last_character(TP_sequences)
TN_sequences = remove_last_character(TN_sequences)
FP_sequences = remove_last_character(FP_sequences)
FN_sequences = remove_last_character(FN_sequences)

print("TP sequences: ", TP_sequences.head(10))
print("TN sequences: ", TN_sequences.head(10))
print("FP sequences: ", FP_sequences.head(10))
print("FN sequences: ", FN_sequences.head(10))

# Define a function to save sequences to a txt file
def save_to_txt(sequences, filename):
    sequences.to_csv(filename, index=False, header=False)

# Save sequences to specified paths
save_to_txt(TP_sequences, "./sequence_to_analysis/Y_TP.txt")
save_to_txt(TN_sequences, "./sequence_to_analysis/Y_TN.txt")
save_to_txt(FP_sequences, "./sequence_to_analysis/Y_FP.txt")
save_to_txt(FN_sequences, "./sequence_to_analysis/Y_FN.txt")

