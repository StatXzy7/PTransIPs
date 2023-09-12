import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import config
import PTransIPs_model
from sklearn.model_selection import KFold, StratifiedShuffleSplit, StratifiedKFold
import collections
from torch.utils.data import  DataLoader, TensorDataset
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import precision_recall_curve,average_precision_score
import os
import pretrained_embedding_generate
from pretrained_embedding_generate import embedding_out
from torch.optim import *
from ml_set import *
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark=True


def get_entropy(probs):
    ent = -(probs.mean(0) * torch.log2(probs.mean(0) + 1e-12)).sum(0, keepdim=True)
    return ent


def get_cond_entropy(probs):
    cond_ent = -(probs * torch.log(probs + 1e-12)).sum(1).mean(0, keepdim=True)
    return cond_ent

def get_val_loss(logits, label, criterion):
    loss = criterion(logits.view(-1, parameters.num_class), label.view(-1))
    loss = (loss.float()).mean()
    loss = (loss - parameters.alpha).abs() + parameters.alpha
    logits = F.softmax(logits, dim=1)  # softmax
  

    sum_loss = loss+get_entropy(logits)-get_cond_entropy(logits)
    return sum_loss[0]


def get_loss(logits, label, criterion):
    loss = criterion(logits.view(-1, parameters.num_class), label.view(-1))
    loss = (loss.float()).mean()
    loss = (loss - parameters.alpha).abs() + parameters.alpha
    return loss

def test_eval(test,test_embedding,test_labels,model):
    Result,_ = model(test,test_embedding)
    Result_softmax = F.softmax(Result, dim=1)  # Apply softmax to the output
    _,predicted=torch.max(Result_softmax,1)
    correct = 0
    correct += (predicted==test_labels).sum().item()
    return 100*correct/Result.shape[0], Result_softmax

def test_eval_str(test,test_embedding,test_str_embedding, test_labels,model):
    Result,_ = model(test,test_embedding, test_str_embedding)
    Result_softmax = F.softmax(Result, dim=1)  # Apply softmax to the output
    _,predicted=torch.max(Result_softmax,1)
    correct = 0
    correct += (predicted==test_labels).sum().item()
    return 100*correct/Result.shape[0], Result_softmax

def addbatch(data,label,batchsize):

    data = TensorDataset(data,label)
    data_loader = DataLoader(data, batch_size=batchsize, shuffle=True)

    return data_loader 

def addbatcht(data,embedding,str_embedding, label,batchsize):

    data = TensorDataset(data,embedding,str_embedding, label)
    data_loader = DataLoader(data, batch_size=batchsize, shuffle=True)

    return data_loader 




def save_model_test(model_dict, fold, auc, save_dir, save_prefix):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # filename = 'test_AUC[{:.3f}], {}.pt'.format(test_auc, save_prefix)
    filename = 'fold{}_BERT_model.pt'.format(fold)
    save_path_pt = os.path.join(save_dir, filename)
    print('save_path_pt',save_path_pt)
    torch.save(model_dict, save_path_pt, _use_new_zipfile_serialization=False)
    print('Save Model Over: {}, AUC: {:.3f}\n'.format(save_prefix, auc))

    

def training(fold, model,device,epochs,criterion,optimizer,
             traindata,
             val,val_embedding,val_str_embedding, val_labels, scheduler):
    
    running_loss = 0
    max_performance = 0
    ReduceLR = False
    model.train()
    for epoch in range(epochs):
        if epoch < warmup_steps:
            scheduler.step()
        elif epoch == warmup_steps:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True, threshold = 1e-3, threshold_mode='abs')
            ReduceLR = True
        for step, (inputs,embedding, str_embedding, labels) in enumerate(traindata):
            # print ("inputs.shape=",inputs.shape)
            inputs = inputs.to(device)
            labels = labels.to(device)
            model = model.to(device)
            outputs,_ = model(inputs, embedding, str_embedding)
            loss = get_val_loss(outputs,labels,criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        acc, val_result = test_eval_str(val,val_embedding,val_str_embedding, val_labels,model)
        auc =  roc_auc_score(val_labels.cpu().detach().numpy(), val_result[:,1].cpu().detach().numpy()) 
        if auc - max_performance > 1e-4 and epoch > 50 :
            print("best_model_save")
            save_model_test(model.state_dict(), fold, auc, './model/Y_train', parameters.learn_name) 
            # Open the file using "append" mode to add content to it
            with open("./model/Y_train/save_result.txt", "a") as f:     
                # Format the content to be written as a string
                result_str = "save model: epoch {} - iteration {}: average loss {:.3f} val_acc {:.3f} val_auc {:.3f} learning rate {:.2e}\n".format(epoch+1, step+1, running_loss,acc,auc, optimizer.param_groups[0]['lr'])
                # Write the string to the file
                f.write(result_str)
            print(result_str, "\n")
            max_performance = auc
            # if epoch >= warmup_steps:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.2
        if epoch%5 == 0:
            print("epoch {} - iteration {}: average loss {:.3f} val_acc {:.3f} val_auc {:.3f} learning rate {:.2e}".format(epoch+1, step+1, running_loss,acc,auc, optimizer.param_groups[0]['lr']))
        running_loss=0
        # if ReduceLR:
        #     scheduler.step(auc)
               


def train_validation(parameters,
                     x_train_encoding,x_train_embedding,x_train_str_embedding,train_label,
                     device,epochs,criterion,k_fold,learning_rate,batchsize):
    # skf = KFold(n_splits=k_fold,shuffle=True,random_state=15)
    skf = StratifiedShuffleSplit(n_splits=k_fold,random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(x_train_encoding,train_label.cpu())):
        model = PTransIPs_model.BERT(parameters).to(device)
        
        print('**'*10,'Fold', fold+1, 'Processing...', '**'*10)
        x_train = x_train_encoding[train_idx]
        x_train_label = train_label[train_idx]
        x_val = x_train_encoding[val_idx]
        x_val_label = train_label[val_idx]
        x_val_label.index = range(len(x_val_label))
        
        embedding = x_train_embedding[train_idx]
        x_val_embedding = x_train_embedding[val_idx]
        str_embedding = x_train_str_embedding[train_idx]
        x_val_str_embedding = x_train_str_embedding[val_idx]
        
        train_data_loader = addbatcht(x_train,embedding,str_embedding, x_train_label,batchsize)
        
        optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
        # optimizer = torch.optim.AdamW(model.parameters(),lr = learning_rate, weight_decay=1e-4)
        warmup_scheduler = WarmupScheduler(optimizer, warmup_steps)
        # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        training(fold,model,device,epochs,criterion,optimizer,
                 train_data_loader,
                 x_val,x_val_embedding,x_val_str_embedding, x_val_label, warmup_scheduler)    
        

def BERT_encoding(txt_array,test_array):
    txt_seq_length = 0
    test_seq_length = 0
    txt_number =len(txt_array)
    test_number = len(test_array)
    for i in range(txt_number):
        txt_array[i] = txt_array[i] + '2'
        if len(txt_array[i]) > txt_seq_length:
            txt_seq_length = len(txt_array[i])
    for i in range(test_number):
        if len(test_array[i]) > test_seq_length:
            test_seq_length = len(test_array[i])
    seq_length = max(txt_seq_length,test_seq_length) 
    x = np.zeros([txt_number, seq_length])
    nuc_d = {'0': 0, '2':2, 'B': 4, 'Q': 5, 'I': 6, 'D': 7, 'M': 8, 'V': 9, 'G': 10, 'K': 11, 'Y': 12, 'P': 13, 'H': 14, 'Z': 15, 'W': 16, 'U': 17, 'A': 18, 'N': 19, 'F': 20, 'R': 21, 'S': 22, 'C': 23, 'E': 24, 'L': 25, 'T': 26, 'X': 27}
    for seq_index in range(txt_number):
        seq = txt_array[seq_index].upper()
        seq = seq[ 0 - seq_length:].ljust(seq_length,'0')
    
        for n, base in enumerate(seq):
            x[seq_index][n] = nuc_d[base]
    x = np.array(x,dtype='int64')
    CLS = np.ones([txt_number,1])
    SEP = 2*np.ones([txt_number,1])
    Pad = np.zeros([txt_number,1])
    x = np.concatenate([CLS,x,Pad,Pad],axis=1) 
    x = torch.LongTensor(x)
    return x

class WarmupScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using {}".format(device))
    
    # ----------------- For S/T dataset ---------------------
    # train,test = data_read()
    # ------------------------------------------------------

    # ----------------- For Y dataset ----------------------
    train,test = data_readY()
    # ------------------------------------------------------

    x_train = train.iloc[:,1]
    x_test = test.iloc[:,1]
    train_label = train.iloc[:,0]
    test_label = test.iloc[:,0]
    
    x_train_encoding = BERT_encoding(x_train,x_test).to('cuda')
    x_test_encoding = BERT_encoding(x_test,x_train).to('cuda')

    # ----------------- For S/T dataset ---------------------
    # x_train_embedding,x_test_embeddin = embedding_load()
    # x_train_str_embedding,x_test_str_embedding = embedding_str_load()
    # ------------------------------------------------------

    # ----------------- For Y dataset ----------------------
    x_train_embedding,x_test_embedding = embedding_loadY()
    x_train_str_embedding,x_test_str_embedding = embedding_str_loadY()
    # ------------------------------------------------------

    train_label = torch.tensor(np.array(train_label,dtype='int64')).to('cuda')
    test_label = torch.tensor(np.array(test_label,dtype='int64')).to('cuda')
    parameters =  config.get_train_config()
    criterion = torch.nn.CrossEntropyLoss()
    
    print("train.shape = ",x_train_encoding.shape)
    print("train_label.shape = ",train_label.shape)
    
    torch.manual_seed(142)
    traindata = addbatch(x_train_encoding,train_label,18)
    device = 'cuda'
    warmup_steps = 5
    train_validation(parameters,
                     x_train_encoding,x_train_embedding,x_train_str_embedding,train_label,
                     device,100,criterion,5,1e-4,15) #New Hyperparameter
    # train_validation(parameters,x_train_encoding,train_label,device,50,criterion,10,0.0001,64)
    #training(model,device,100,criterion,optimizer,traindata,x_test_encoding,test_label)
    