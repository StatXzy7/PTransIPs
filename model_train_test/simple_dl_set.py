# -*- coding: utf-8 -*-
"""
@author: ZiyangXu
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from argparse import Namespace
from torch.utils.data import  DataLoader, TensorDataset
import torch.nn.functional as F
import tqdm
from tqdm import *



def data_read():
    train = pd.read_csv('../data/DeepIPS_Train_data.csv',header=0)
    test = pd.read_csv('../data/DeepIPS_Test_data.csv',header=0)
    # train = pd.read_csv('../data/Homo_train_select.csv',header=0)
    # test = pd.read_csv('../data/Homo_Test_select.csv',header=0)
    return train,test



args = Namespace(
    num_vocab = 200,
    embedding_dim = 100,
    hidden_size = 20,
    num_layers = 3,
)

class LSTM_one_hot(nn.Module):
    
    def __init__(self):
        super(LSTM_one_hot,self).__init__()
        #self.embedding =nn.Embedding(args.num_vocab, args.embedding_dim,max_norm=1,norm_type=2)
        self.rnn = torch.nn.LSTM(21, hidden_size=args.hidden_size,num_layers=args.num_layers,batch_first=True,dropout=0.5)
        self.fc1 = torch.nn.Linear(args.hidden_size*33, 10)
        self.fc2 = torch.nn.Linear(10,2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self,seq_in):
        embed  = F.one_hot(seq_in)
        embed = embed.float()
        h1,(h_n,h_c) = self.rnn(embed)
        h1 = torch.flatten(h1,start_dim=1,end_dim=2)
        linear1 = self.fc1(self.relu(h1))
        out = torch.nn.functional.softmax(self.fc2(linear1))
        return out

class LSTM_embed(nn.Module):
    
    def __init__(self):
        super(LSTM_embed,self).__init__()
        self.embedding =nn.Embedding(args.num_vocab, args.embedding_dim,max_norm=1,norm_type=2)
        self.rnn = torch.nn.LSTM(args.embedding_dim, hidden_size=args.hidden_size,num_layers=args.num_layers,batch_first=True,dropout=0.5)
        self.fc1 = torch.nn.Linear(args.hidden_size*33, 32)
        self.fc2 = torch.nn.Linear(32,16)
        self.fc3 = torch.nn.Linear(16,2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,seq_in):
        embed  = self.embedding(seq_in)
        h1,(h_n,h_c) = self.rnn(embed)
        h1 = torch.flatten(h1,start_dim=1,end_dim=2)
        linear1 = self.fc1(self.relu(h1))
        linear2 = self.fc2(self.relu(linear1))
        out = torch.nn.functional.softmax(self.fc3(linear2))
        return out
    
DPargs = Namespace(
    num_vocab=1000,
    embedding_dim=128,
    padding_index=0,
    region_embedding_size=3,             # Region embedding kernel size
    cnn_num_channel=256,
    cnn_kernel_size=3,              # CNN参数应保证为等长卷积
    cnn_padding_size=1,
    cnn_stride=1,
    pooling_size=2,         # MaxPooling应保证resize为原1/2
    num_classes=2,
)


class DPCNN(nn.Module):
    def __init__(self):
        super(DPCNN, self).__init__()
        self.embedding = nn.Embedding(DPargs.num_vocab, DPargs.embedding_dim, padding_idx=DPargs.padding_index)
        self.region_cnn = nn.Conv1d(DPargs.embedding_dim, DPargs.cnn_num_channel, DPargs.region_embedding_size)
        self.padding1 = nn.ConstantPad1d((1, 1), 0)          # region embedding后的对齐
        self.padding2 = nn.ConstantPad1d((0, 1), 0)       # block2中先补齐，先防止信息丢失
        self.relu = nn.ReLU()
        self.cnn = nn.Conv1d(DPargs.cnn_num_channel, DPargs.cnn_num_channel, kernel_size=DPargs.cnn_kernel_size,
                             padding=DPargs.cnn_padding_size, stride=DPargs.cnn_stride)
        self.maxpooling = nn.MaxPool1d(kernel_size=DPargs.pooling_size)
        self.fc = nn.Linear(DPargs.cnn_num_channel, DPargs.num_classes)

    def forward(self, x_in):
        emb = self.embedding(x_in)           # Batch*Sequence*Embedding
        emb = self.region_cnn(emb.transpose(1, 2))          # Batch*Embedding*Sequence——>Batch*Embedding*(Sequence-2)
        emb = self.padding1(emb)         # Batch*Embedding*(Sequence-2)——> Batch*Embedding*Sequence

        # 第一层的卷积+skip-connection
        conv = emb + self._block1(self._block1(emb))

        # 第二层的block+skip-connection
        while conv.size(-1) >= 2:
            conv = self._block2(conv)
        
        out = torch.nn.functional.softmax(self.fc(torch.squeeze(conv, dim=-1)))
        return out

    def _block1(self, x):
        return self.cnn(self.relu(x))          # 注意这里是pre-activation

    def _block2(self, x):
        x = self.padding2(x)
        px = self.maxpooling(x)
        x = self.relu(px)
        x = self.cnn(x)
        x = self.relu(x)
        x = self.cnn(x)
        x = px + x
        return x

def get_attn_pad_mask(seq):
    batch_size, seq_len = seq.size()
    pad_attn_mask = seq.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    pad_attn_mask_expand = pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]
    return pad_attn_mask_expand


class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding (look-up table) [64, 43, 64]
        self.pos_embed = nn.Embedding(max_len, d_model)  # position embedding [64, 43, 64]
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)  # x: [batch_size, seq_len]
        pos = torch.arange(seq_len, device=device, dtype=torch.long)  # [seq_len]
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        embedding = self.pos_embed(pos)
        embedding = embedding + self.tok_embed(x)
        embedding = self.norm(embedding)
        return embedding


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_head, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_head, seq_len, seq_len]
        context = torch.matmul(attn, V)  # [batch_size, n_head, seq_len, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_head)
        self.W_K = nn.Linear(d_model, d_k * n_head)
        self.W_V = nn.Linear(d_model, d_v * n_head)

        self.linear = nn.Linear(n_head * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # q_s: [batch_size, n_head, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # k_s: [batch_size, n_head, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_head, d_v).transpose(1, 2)  # v_s: [batch_size, n_head, seq_len, d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_head, 1, 1)
        context, attention_map = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_head * d_v)  # context: [batch_size, seq_len, n_head * d_v]
        output = self.linear(context)
        output = self.norm(output + residual)
        return output, attention_map


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.attention_map = None

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attention_map = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                                        enc_self_attn_mask)  # enc_inputs to same Q,K,V
        self.attention_map = attention_map
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs


class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()

        global max_len, n_layers, n_head, d_model, d_ff, d_k, d_v, vocab_size, device
        max_len = config.max_len
        n_layers = config.num_layer
        n_head = config.num_head
        d_model = config.dim_embedding
        d_ff = config.dim_feedforward
        d_k = config.dim_k
        d_v = config.dim_v
        vocab_size = config.vocab_size
        device = torch.device("cuda" if config.cuda else "cpu")

        self.embedding = Embedding(config)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc_task = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),
        )
        self.classifier = nn.Linear(2, 2)

    def forward(self, input_ids):
        output = self.embedding(input_ids)  # [bach_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids)  # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)
            # output: [batch_size, max_len, d_model]

        # classification
        # only use [CLS]
        representation = output[:, 0, :]
        reduction_feature = self.fc_task(representation)
        reduction_feature = reduction_feature.view(reduction_feature.size(0), -1)
        logits_clsf = self.classifier(reduction_feature)
        representation = reduction_feature
        return logits_clsf, representation

#one_hot_train = nn.functional.one_hot(torch.squeeze(torch_train))
#embedding  = nn.Embedding(30, 10,max_norm=1,norm_type=2)
#Embedding_train = embedding(torch_train)
"""
for epoch in range(5):
    model.train()
    inputs = torch_train
    labels = train_label
    labels = torch.unsqueeze(labels,1)
    outputs = model(inputs)
    loss = criterion(outputs,labels)
    loss.backward()
    optimizer.step()
"""
def encoding_first(txt_array):
    seq_length = 0
    number = len(txt_array)
    for i in range(number):
        if len(txt_array[i]) > seq_length:
            seq_length = len(txt_array[i])
    x = np.zeros([number, seq_length])
    nuc_d = {'0':0,'A':1,'B':22,'U':23,'J':24,'Z':25,'O':26,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,'X':21}
    for seq_index in range(number):
        seq = txt_array[seq_index].upper()
        seq = seq[ 0 - seq_length:].ljust(seq_length,'0')
    
        for n, base in enumerate(seq):
            x[seq_index][n] = nuc_d[base]
    x = np.array(x,dtype='int64')
    x = torch.LongTensor(x)
    return x

def BERT_encoding(txt_array,test_array):
    txt_seq_length = 0
    test_seq_length = 0
    txt_number =len(txt_array)
    test_number = len(test_array)
    for i in range(txt_number):
        txt_array[i] = txt_array[i] + '2'
    for i in range(txt_number):
        if len(txt_array[i]) > txt_seq_length:
            txt_seq_length = len(txt_array[i])
    for i in range(test_number):
        if len(test_array[i]) > test_seq_length:
            test_seq_length = len(test_array[i])
    seq_length = max(txt_seq_length,test_seq_length)
    x = np.zeros([txt_number, seq_length])
    nuc_d = {'0': 0, '[CLS]': 1, '2': 2, '[MASK]': 3, 'B': 4, 'Q': 5, 'I': 6, 'D': 7, 'M': 8, 'V': 9, 'G': 10, 'K': 11, 'Y': 12, 'P': 13, 'H': 14, 'Z': 15, 'W': 16, 'U': 17, 'A': 18, 'N': 19, 'F': 20, 'R': 21, 'S': 22, 'C': 23, 'E': 24, 'L': 25, 'T': 26, 'X': 27}
    for seq_index in range(txt_number):
        seq = txt_array[seq_index].upper()
        seq = seq[ 0 - seq_length:].ljust(seq_length,'0')
    
        for n, base in enumerate(seq):
            x[seq_index][n] = nuc_d[base]
    x = np.array(x,dtype='int64')
    CLS = np.ones([txt_number,1])
    #SEP = 2*np.ones([txt_number,1])
    Pad = np.zeros([txt_number,1])
    x = np.concatenate([CLS,x,Pad,Pad],axis=1)
    x = torch.LongTensor(x)
    return x


def addbatch(data_train,data_test,batchsize):
    """
    设置batch
    :param data_train: 输入
    :param data_test: 标签
    :param batchsize: 一个batch大小
    :return: 设置好batch的数据集
    """
    data = TensorDataset(data_train,data_test)
    data_loader = DataLoader(data, batch_size=batchsize, shuffle=True)#shuffle是是否打乱数据集，可自行设置

    return data_loader


def test_eval(test,test_labels,model):
    Result = model(test)
    _,predicted=torch.max(Result,1)
    correct = 0
    correct += (predicted==test_labels).sum().item()
    #for i in range(Result.shape[0]):
       #if predicted[i] == test_labels[i]:
          #correct = correct+1
    return 100*correct/Result.shape[0]



