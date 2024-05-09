# ---encoding:utf-8---

import os,sys 
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir) 



import torch
import torch.nn as nn
import numpy as np
import config

config = config.get_train_config()

def get_attn_pad_mask(seq):
    # print("seq.size = ", seq.size())
    batch_size, seq_len = seq.size()
    pad_attn_mask = seq.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    pad_attn_mask_expand = pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]
    # print("pad_attn_mask_expand.shape = ",pad_attn_mask_expand.shape)
    return pad_attn_mask_expand


class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d0_model)  # token embedding (look-up table) [64, 43, 64]
        self.pos_embed = nn.Embedding(max_len, d0_model)  # position embedding [64, 43, 64]
        self.norm = nn.LayerNorm(d0_model)

    def forward(self, x):
        seq_len = x.size(1)  # x: [batch_size, seq_len]
        pos = torch.arange(seq_len, device=device, dtype=torch.long)  # [seq_len]
        
        # pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        pos = torch.arange(seq_len, device=device, dtype=torch.long)
        pos = pos.unsqueeze(0).repeat(x.size(0), 1)  # [seq_len] -> [batch_size, seq_len]

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
        # self.dropout = nn.Dropout(config.dropout)  # 添加dropout层

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0) # Q,K,V [64,33,1156]
        # residual, batch_size = Q, Q.shape(0) # Q,K,V [64,33,1156]
        # print("Q.shape",Q.shape)
        q_s = self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # q_s: [batch_size, n_head, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # k_s: [batch_size, n_head, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_head, d_v).transpose(1, 2)  # v_s: [batch_size, n_head, seq_len, d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_head, 1, 1)
        context, attention_map = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_head * d_v)  # context: [batch_size, seq_len, n_head * d_v]
        output = self.linear(context)
        output = self.norm(output + residual)
        # self.dropout = nn.Dropout(config.dropout)  # 添加dropout层
        
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
        self.dropout = nn.Dropout(config.dropout)  # 添加dropout层

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attention_map = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                                        enc_self_attn_mask)  # enc_inputs to same Q,K,V [64,33,1156]
        self.attention_map = attention_map
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        enc_outputs = self.dropout(enc_outputs)  # 添加dropout操作
        return enc_outputs

class ResNetConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ResNetConv1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()

        global max_len, n_layers, n_head, d0_model, d_model, d_ff, d_k, d_v, vocab_size, device
        max_len = config.max_len
        n_layers = config.num_layer
        n_head = config.num_head
        d0_model = config.dim_embedding
        # d_model = config.dim_embedding 
        d_model = config.dim_embedding + 256
        d_ff = config.dim_feedforward
        d_k = config.dim_k
        d_v = config.dim_v
        vocab_size = config.vocab_size
        if config.task == 'test':
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if config.cuda else "cpu")

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model , config.num_head , config.dim_feedforward , dropout = 0.1)
            for _ in range(config.num_layer)
        ])
        self.embedding = Embedding(config)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        
        self.conv1dEmbed = nn.Sequential(
            # 这个函数用来把预训练embedding的维度从33变成37
            nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=4), 
            nn.ReLU(), # 86
        )
                
        self.conv1dStr = nn.Sequential(
            nn.Conv1d(132, 256, kernel_size=5, stride=1, padding=4),
            # nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        
        self.resconv1d = nn.Sequential(
            ResNetConv1d(in_channels=d_model, out_channels=d_model, kernel_size=5, stride=1, padding=2), #这里的1156就是输入维度
            nn.Dropout(0.5),
            nn.ReLU()
        )
        
        self.fc_task = nn.Sequential(
            nn.Linear(2*d_model, 2*d_model // 2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(2*d_model // 2, 2),
        )
        # self.fc_task = nn.Sequential(
        #     nn.Linear(256, 256 // 2),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(256 // 2, 2),
        # )

        self.classifier = nn.Linear(2, 2)



    def forward(self, input_ids, x_embedding, str_embedding):
        
        # print("input_ids.shape=",input_ids.shape)
        # print("str_embedding.shape", str_embedding.shape)
        # print("str_embedding.type", str_embedding.dtype)
        str_embedding = str_embedding.to(torch.float32) #str_embedding是float64精度的，需要修改 [64,33,132]
        # str_embedding = str_embedding[:, :, 33:66].to(torch.float32) #只提取第三个维度的33:66，刚好是平均距离的str_embedding
        

        # You need to permute the dimensions before passing it to the conv layer
        str_embedding = str_embedding.permute(0, 2, 1)
        output_data = self.conv1dStr(str_embedding)
        # Now permute it back to [batch_size, seq_len, feature_size]
        str_embedding = output_data.permute(0, 2, 1)
        # print(str_embedding.shape)  # Should be torch.Size([64, 33, 256])

        # Embedding 通过1dCNN维度调整
        # encoding中需要补0，所以还是需要再cnn调整成37 [64, 33, 1024]
        x_embedding = x_embedding.permute(0, 2, 1) # 调整维度顺序输入cnn (batch_size, in_channels, seq_length)
        x_embedding = self.conv1dEmbed(x_embedding)
        x_embedding = x_embedding.permute(0, 2, 1)  #[64,33,1024] 33->37  [64,37,1024]
        # print("x_embedding_conv.shape=",x_embedding.shape) 


        
        #并行的Transformer
        
        #这里的self.embedding是单纯的独热编码
        self_embedding = self.embedding(input_ids)  # [bach_size, seq_len, d_model = dim_embedding] [64,33,1024]
        # print("self_embedding.shape", self_embedding.shape)
        # print("self_embedding.type", self_embedding.dtype)
        
        # all_input = self_embedding + x_embedding
        all_input = torch.cat((self_embedding + x_embedding , str_embedding), dim=2) # [64,33,1156]
        # print("all_input.shape", all_input.shape)
        
        enc_self_attn_mask = get_attn_pad_mask(input_ids)  # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            output_t = layer(all_input, enc_self_attn_mask)  # 64,37,1156
        # print("output.shape=",output.shape)  
        
        #并行的CNN
        output_c = self.resconv1d((all_input).permute(0, 2, 1))
        output_c = output_c.permute(0, 2, 1)
        
        #concat Transformer和CNN
        output = torch.cat([output_t, output_c], dim=2) # torch.Size([64, 37, 1024])
        # print("output.shape=",output.shape) 
        
        # classification
        # only use [CLS]
        representation = output[:, 0, :]    
        reduction_feature = self.fc_task(representation)
        reduction_feature = reduction_feature.view(reduction_feature.size(0), -1)
        logits_clsf = self.classifier(reduction_feature)
        # representation = reduction_feature
        # print("reduction_feature.shape0",representation.shape)
        return logits_clsf,representation



