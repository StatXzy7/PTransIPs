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
from umap_model import *
from train import BERT_encoding
from Algorithm_eval_function import LR_eval,LR_L2_eval,Linear_SVM_eval,Kernel_SVM_eval,RF_eval,DL_eval,BERT_eval,preBERT_eval
import config

import umap

dataset = 'trainset'
# dataset = 'testset'

cf = config.get_train_config()
cf.task = 'test'
device = torch.device('cpu')
BERT_model = BERT(cf)

# -----------------ST---------------------
train,test = data_read()
train_seq = train.iloc[:,1]
test_seq = test.iloc[:,1]
train_label = torch.tensor(np.array(train.iloc[:,0],dtype='int64')).to(device)
test_label = torch.tensor(np.array(test.iloc[:,0],dtype='int64')).to(device) #Important
train_encoding = BERT_encoding(train_seq,test_seq)
test_encoding = BERT_encoding(test_seq,train_seq)
train_embedding = torch.tensor(np.load('./embedding/x_train_embedding.npy')).to(device)
test_embedding = torch.tensor(np.load('./embedding/x_test_embedding.npy')).to(device)
train_str_embedding = torch.tensor(np.load('./embedding/train_str_embedding.npy')).to(device)
test_str_embedding = torch.tensor(np.load('./embedding/test_str_embedding.npy')).to(device) 

path = './model/ST'
#-------------------Please change the file path here to fit your model---------------
pt_file = './model/ST/ST_model.pt'

print("model loading......")
BERT_model.load_state_dict(torch.load(pt_file))
BERT_model = BERT_model.to(device)
BERT_model.eval()

if dataset == 'trainset':
    input_ids, self_embedding, x_embedding, str_embedding, representation, logits_clsf = BERT_model(train_encoding,train_embedding, train_str_embedding) # ST_train
    print("trainset model loaded!")
else:
    input_ids, self_embedding, x_embedding, str_embedding, representation, logits_clsf = BERT_model(test_encoding,test_embedding, test_str_embedding) # ST_test
    print("testset model loaded!")

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
if dataset == 'trainset':
    test_label_np = train_label.cpu().detach().numpy() # ST train
else:
    test_label_np = test_label.cpu().detach().numpy() # ST test

logits_clsf_np = logits_clsf.cpu().detach().numpy()

# Flatten x_embedding, self_embedding, str_embedding from 3D to 2D
print("Flattening x_embedding, self_embedding, str_embedding...")
x_embedding_np = x_embedding_np.reshape(x_embedding_np.shape[0], -1)
self_embedding_np = self_embedding_np.reshape(self_embedding_np.shape[0], -1)
str_embedding_np = str_embedding_np.reshape(str_embedding_np.shape[0], -1)

# Create a dictionary to iterate over
print("Creating data dictionary...")
# data_dict = {'Raw input data': input_ids_np, 
#              'Token and Position embedding': self_embedding_np,
#              'Protein pretrained language model sequence embedding': x_embedding_np, 
#              'Protein pretrained language model structure embedding': str_embedding_np,
#              'Representation after CNN and transformer': representation_np,
#              'Final classification': logits_clsf_np
#              }
data_dict = {'Raw input data': input_ids_np, 
             'Token and Position embedding': self_embedding_np,
             'Pretrained sequence embedding': x_embedding_np, 
             'Pretrained structure embedding': str_embedding_np,
             'Representation after Transformer': representation_np,
             'Final classification': logits_clsf_np
             }

# Create a UMAP reducer
print("Creating UMAP reducer...")
reducer = umap.UMAP()

# Create the directory for saving the figures
print("Creating directory for figures...")
os.makedirs("./figures/umap_pdf", exist_ok=True)

colors = [(0.3, 0.6, 0.9, 1), (1, 0, 0, 0.5)]   # blue, red

for name, data in data_dict.items():
    print(f"Processing {name}...")

    # Use UMAP to reduce the dimensionality of your data to 2
    print("Reducing data dimensionality with UMAP...")
    data_reduced = reducer.fit_transform(data)

    # Plot the reduced data, coloring by label
    print("Plotting data...")
    plt.figure(figsize=(12, 12))
    
    # Create separate scatter plots for positive and negative labels
    plt.scatter(data_reduced[test_label_np == 1, 0], data_reduced[test_label_np == 1, 1], 
                c=colors[0], label='positive')
    plt.scatter(data_reduced[test_label_np == 0, 0], data_reduced[test_label_np == 0, 1], 
                c=colors[1], label='negative')
    
    # Use the modified title with increased font size
    if dataset == 'trainset':
        plt.title(f'UMAP of S/T train for \n {name}', fontsize=32)
    else:
        plt.title(f'UMAP of S/T test for \n {name}', fontsize=32)

    # Add x and y axis labels with increased font size
    plt.xlabel('UMAP Dimension 1', fontsize=26)
    plt.ylabel('UMAP Dimension 2', fontsize=26)

    # Increase the font size of axis ticks
    plt.tick_params(axis='both', labelsize=24)

    # Add a legend with increased font size
    plt.legend(fontsize=20)
    
    # Save the figure
    print("Saving figure...")
    safe_name = name.replace(" ", "_")  # replace ' ' in name to '_'
    if dataset == 'trainset':
        path = f"./figures/umap_pdf/umap_ST_train_{safe_name}.pdf" #ST_train
    else:
        path = f"./figures/umap_pdf/umap_ST_test_{safe_name}.pdf" #ST_test
    plt.savefig(path)

    print(f"{name} processing complete.\n")

print("All processing complete.")


