# The running time of this code is relatively long

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
from BERT_model_pretrain_final_eval import *
from bert_train_pretrain_final import BERT_encoding


from Algorithm_eval_function import LR_eval,LR_L2_eval,Linear_SVM_eval,Kernel_SVM_eval,RF_eval,DL_eval,BERT_eval,preBERT_eval
import pickle
import ml_config
import config

import umap

cf = config.get_train_config()
cf.task = 'test'
device = torch.device('cpu')
BERT_model = BERT(cf)

# -----------------Y---------------------
train,test = data_readY()
train_seq = train.iloc[:,1]
test_seq = test.iloc[:,1]
train_label = torch.tensor(np.array(train.iloc[:,0],dtype='int64')).to(device)
test_label = torch.tensor(np.array(test.iloc[:,0],dtype='int64')).to(device) #非常重要
train_encoding = BERT_encoding(train_seq,test_seq)
test_encoding = BERT_encoding(test_seq,train_seq)
train_embedding = torch.tensor(np.load('/root/autodl-tmp/myDNAPredict/program 1.1/data/Y_train_embedding.npy')).to(device)
test_embedding = torch.tensor(np.load('/root/autodl-tmp/myDNAPredict/program 1.1/data/Y_test_embedding.npy')).to(device)
train_str_embedding = torch.tensor(np.load('/root/autodl-tmp/myDNAPredict/program 1.1/data/Y_train_str_embedding.npy')).to(device)
test_str_embedding = torch.tensor(np.load('/root/autodl-tmp/myDNAPredict/program 1.1/data/Y_test_str_embedding.npy')).to(device) 

path = '/root/autodl-tmp/myDNAPredict/program 1.1/dl_model_Y'
pt_file = '/root/autodl-tmp/myDNAPredict/program 1.1/dl_model_Y/fold8_BERT_model.pt'

print("model loading......")
BERT_model.load_state_dict(torch.load(pt_file))
BERT_model = BERT_model.to(device)
BERT_model.eval()

# input_ids, self_embedding, x_embedding, str_embedding, representation, logits_clsf = BERT_model(test_encoding,test_embedding, test_str_embedding) # Y_test
input_ids, self_embedding, x_embedding, str_embedding, representation, logits_clsf = BERT_model(train_encoding,train_embedding, train_str_embedding) # Y_train
print("model loaded!")

print("input_ids.shape = ", input_ids.shape)
print("self_embedding.shape = ", self_embedding.shape)
print("x_embedding.shape = ", x_embedding.shape)
print("str_embedding.shape = ", str_embedding.shape)
print("representation.shape = ", representation.shape)
print("test_label.shape = ", test_label.shape)
print("logits_clsf.shape = ", logits_clsf.shape)

# Assuming input_ids, self_embedding, x_embedding, str_embedding, and representation are your tensors,
# and test_label is your labels tensor. Convert them to numpy arrays:
print("Converting tensors to numpy arrays...")
input_ids_np = input_ids.cpu().detach().numpy()
self_embedding_np = self_embedding.cpu().detach().numpy()
x_embedding_np = x_embedding.cpu().detach().numpy()
str_embedding_np = str_embedding.cpu().detach().numpy()
representation_np = representation.cpu().detach().numpy()
# test_label_np = test_label.cpu().detach().numpy() # Y test
test_label_np = train_label.cpu().detach().numpy() # Y train
logits_clsf_np = logits_clsf.cpu().detach().numpy()

# Flatten x_embedding, self_embedding, str_embedding from 3D to 2D
print("Flattening x_embedding, self_embedding, str_embedding...")
x_embedding_np = x_embedding_np.reshape(x_embedding_np.shape[0], -1)
self_embedding_np = self_embedding_np.reshape(self_embedding_np.shape[0], -1)
str_embedding_np = str_embedding_np.reshape(str_embedding_np.shape[0], -1)

# Create a dictionary to iterate over
print("Creating data dictionary...")
data_dict = {'Raw input data': input_ids_np, 
             'Token and Position embedding': self_embedding_np,
             'Protein pretrained language model sequence embedding': x_embedding_np, 
             'Protein pretrained language model structure embedding': str_embedding_np,
             'Representation after CNN and transformer': representation_np,
             'Final classification': logits_clsf_np
             }

# Create a UMAP reducer
print("Creating UMAP reducer...")
reducer = umap.UMAP()

# Create the directory for saving the figures
print("Creating directory for figures...")
os.makedirs("/root/autodl-tmp/myDNAPredict/program 1.1/figures/umap", exist_ok=True)

for name, data in data_dict.items():
    print(f"Processing {name}...")

    # Use UMAP to reduce the dimensionality of your data to 2
    print("Reducing data dimensionality with UMAP...")
    data_reduced = reducer.fit_transform(data)

    # Plot the reduced data, coloring by label
    print("Plotting data...")
    plt.figure(figsize=(10, 10))
    
    # Create separate scatter plots for positive and negative labels # Y 
    plt.scatter(data_reduced[test_label_np == 1, 0], data_reduced[test_label_np == 1, 1], 
                c='blue', label='positive')
    plt.scatter(data_reduced[test_label_np == 0, 0], data_reduced[test_label_np == 0, 1], 
                c='orange', label='negative')
    
    # plt.title(f'UMAP visualization of Y test for {name}') #Y_test
    plt.title(f'UMAP visualization of Y train for {name}') #Y_train

    # Add a legend
    plt.legend()

    # Save the figure
    print("Saving figure...")
    # path = f"/root/autodl-tmp/myDNAPredict/program 1.1/figures/umap/umap_Y_test_{name}.png" #Y_test
    path = f"/root/autodl-tmp/myDNAPredict/program 1.1/figures/umap/umap_Y_train_{name}.png" #Y_train

    plt.savefig(path)

    print(f"{name} processing complete.\n")

print("All processing complete.")

