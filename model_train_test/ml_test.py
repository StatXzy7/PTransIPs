# -*- coding: utf-8 -*-
"""
@author: ZiyangXu
"""

import numpy as  np
import pandas as pd
import sklearn
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark=True
import torch.nn.functional as F
from ml_set import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,roc_auc_score
from simple_dl_set import *
#from dl_test import *
from sklearn.model_selection import KFold, StratifiedShuffleSplit, StratifiedKFold


import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train,test = data_read()
train_seq = train.iloc[:,1]
test_seq = test.iloc[:,1]
train_label = train.iloc[:,0]
test_label = test.iloc[:,0]
dl_train_encoding = encoding_first(train_seq).to(device)
dl_test_encoding = encoding_first(test_seq).to(device)
dl_train_label = torch.tensor(np.array(train.iloc[:,0],dtype='int64')).to(device)
dl_test_label = torch.tensor(np.array(test.iloc[:,0],dtype='int64')).to(device)


def save_model(model_dict, best_acc, save_dir, save_prefix):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = 'ACC[{:.4f}], {}.pt'.format(best_acc, save_prefix)
    save_path_pt = os.path.join(save_dir, filename)
    print('save_path_pt',save_path_pt)
    torch.save(model_dict, save_path_pt, _use_new_zipfile_serialization=False)
    print('Save Model Over: {}, ACC: {:.4f}\n'.format(save_prefix, best_acc))





def DPCNN_training(model,epochs,criterion,optimizer,traindata,test,test_labels):
    running_loss = 0
    max_performance = 0
    for epoch in range(epochs):
        for step, (inputs,labels) in enumerate(traindata):
            inputs = inputs.to(device)
            labels = labels.to(device)
            model = model.cuda()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step % 8 == 0:
                acc = test_eval(test,test_labels,model)
                if acc > max_performance and acc > 80:
                    #torch.save(model.state_dict(),'../dl_model_save/best_save.pt')
                    save_model(model.state_dict(), acc, '../dl_model_save', 'DPCNN')
                    print("best_model_save")
                    max_performance = acc
                   
                print("epoch {} - iteration {}: average loss {:.3f} val_accuracy {:.3f}".format(epoch+1, step+1, running_loss,acc))
            running_loss=0



def DPCNN_train_validation(x_train_encoding,train_label,device,epochs,criterion,k_fold,learning_rate,batchsize):
    skf = KFold(n_splits=k_fold,shuffle=True,random_state=15)
    for fold, (train_idx, val_idx) in enumerate(skf.split(x_train_encoding,train_label)):
        model = DPCNN().to(device)
        print('**'*10,'第', fold+1, '折','ing....', '**'*10)
        x_train = x_train_encoding[train_idx]
        x_train_label = train_label[train_idx]
        x_val = x_train_encoding[val_idx]
        x_val_label = train_label[val_idx]
        x_val_label.index = range(len(x_val_label))
        train_data_loader = addbatch(x_train,x_train_label,batchsize)
        optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
        DPCNN_training(model,epochs,criterion,optimizer,train_data_loader,x_val,x_val_label) 




def LSTM_training(model,epochs,criterion,optimizer,traindata,test,test_labels):
    running_loss = 0
    max_performance = 0
    for epoch in range(epochs):
        for step, (inputs,labels) in enumerate(traindata):
            inputs = inputs.to(device)
            labels = labels.to(device)
            model = model.cuda()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step % 8 == 0:
                acc = test_eval(test,test_labels,model)
                if acc > max_performance and acc > 80:
                    #torch.save(model.state_dict(),'../dl_model_save/best_save.pt')
                    save_model(model.state_dict(), acc, '../dl_model_save', 'LSTM')
                    print("best_model_save")
                    max_performance = acc
                   
                print("epoch {} - iteration {}: average loss {:.3f} val_accuracy {:.3f}".format(epoch+1, step+1, running_loss,acc))
            running_loss=0



def LSTM_train_validation(x_train_encoding,train_label,device,epochs,criterion,k_fold,learning_rate,batchsize):
    skf = KFold(n_splits=k_fold,shuffle=True,random_state=15)
    for fold, (train_idx, val_idx) in enumerate(skf.split(x_train_encoding,train_label)):
        model = LSTM_embed().to(device)
        print('**'*10,'第', fold+1, '折','ing....', '**'*10)
        x_train = x_train_encoding[train_idx]
        x_train_label = train_label[train_idx]
        x_val = x_train_encoding[val_idx]
        x_val_label = train_label[val_idx]
        x_val_label.index = range(len(x_val_label))
        train_data_loader = addbatch(x_train,x_train_label,batchsize)
        optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
        LSTM_training(model,epochs,criterion,optimizer,train_data_loader,x_val,x_val_label) 

LSTM_model = LSTM_embed()
LSTM_model = LSTM_embed().to(device)
DPCNN_model = DPCNN()
DPCNN_model = DPCNN_model.to(device)

criterion = torch.nn.CrossEntropyLoss()
batchsize = 64
if __name__ == '__main__':
    DPCNN_train_validation(dl_train_encoding,dl_train_label,device,30,criterion,10,0.0001,64)
    LSTM_train_validation(dl_train_encoding,dl_train_label,device,100,criterion,10,0.0001,64)


