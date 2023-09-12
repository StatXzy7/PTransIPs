from transformers import T5EncoderModel, T5Tokenizer
import torch
import numpy as np
import pandas as pd
from ml_set import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_max_len_index(data):
    train_length_frame = data.str.len()
    max_len_index = train_length_frame.argmax()
    return max_len_index
    

def str_padding(txt_array,test_array):
    txt_seq_length = 0
    test_seq_length = 0
    txt_number =len(txt_array)
    test_number = len(test_array)
    data = txt_array
    for i in range(txt_number):
        if len(data[i]) > txt_seq_length:
            txt_seq_length = len(data[i])
    for i in range(test_number):
        if len(test_array[i]) > test_seq_length:
            test_seq_length = len(test_array[i])
    seq_length = max(txt_seq_length,test_seq_length)
    for seq_index in range(txt_number):
        seq= data[seq_index].upper()
    
        data[seq_index] = seq[ 0 - seq_length:].ljust(seq_length,'0')
    return data

def make_fasta_rough(data,max_len_index):
    short = list([])   
    for i in range(len(data)):
        short.append(str(data.iloc[i]))
    short = pd.DataFrame(short)
    short_seq = short.iloc[:,0]
    short_fasta=[]
    for i in range(len(short_seq)):
        short_fasta.append('>'+str(5000000+i))
        short_fasta.append(short_seq.iloc[max_len_index])
        short_fasta.append('>'+str(i))
        short_fasta.append(short_seq.iloc[i])
    short_fasta=pd.DataFrame(short_fasta)
    return short_fasta


def read_fasta( fasta_f, split_char="!", id_field=0):    
    seqs = dict()   
    for line in fasta_f[0]:
        # get uniprot ID from header and create new entry
        if line.startswith('>'):
            uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
            # replace tokens that are mis-interpreted when loading h5
            uniprot_id = uniprot_id.replace("/","_").replace(".","_")
            seqs[ uniprot_id ] = ''
        else:
            # repl. all whie-space chars and join seqs spanning multiple lines, drop gaps and cast to upper-case
            seq= ''.join( line.split() ).upper().replace("-","")
            # repl. all non-standard AAs and map them to unknown/X
            seq = seq.replace('U','X').replace('Z','X').replace('O','X')
            seqs[ uniprot_id ] += seq 
    # example_id=next(iter(seqs))
    # print("Read {} sequences.".format(len(seqs)))
    # print("Example:\n{}\n{}".format(example_id,seqs[example_id]))

    return seqs


def get_T5_model():
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    return model, tokenizer



def get_embeddings( model, tokenizer, seqs, per_residue,
                    max_residues, max_seq_len, max_batch):


    results = {"residue_embs" : dict() }

    seq_dict   = sorted( seqs.items(), key=lambda kv: len( seqs[kv[0]] ), reverse=True )
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict,1):
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id,seq,seq_len))

        n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len 
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # add_special_tokens adds extra token at the end of each sequence
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True,padding=True)
            input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)          
            try:
                with torch.no_grad():
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            for batch_idx, identifier in enumerate(pdb_ids): # for each protein in the current mini-batch
                s_len = seq_lens[batch_idx]
                emb = embedding_repr.last_hidden_state[batch_idx,:s_len]
                if per_residue: # store per-residue embeddings (Lx1024)
                    results["residue_embs"][ identifier ] = emb.detach().cpu().numpy().squeeze()
    return results


def embedding_out(train_data,test_data,per_residue,max_residues=400, max_seq_len=100, max_batch=100,path=False):
    model, tokenizer = get_T5_model()
    max_len_index = get_max_len_index(train_data)
    train = str_padding(train_data,test_data)
    train = make_fasta_rough(train_data,max_len_index)
    # Load example fasta.
    train = read_fasta(train)  
    # Compute embeddings and/or secondary structure predictions
    results = get_embeddings( model, tokenizer, train,per_residue,max_residues, max_seq_len, max_batch)
    model.cpu()
    del model
    dic = results["residue_embs"]
    results_dic = {key:dic[key] for key in dic.keys() if int(key) < 5000000}
    
    final_embedding = np.array(list(results_dic.values()))
    return final_embedding

if __name__ == "__main__":
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

    data = pd.concat([x_train, x_test])
    labels = pd.concat([train_label, test_label])   

    print("x_train.shape= ", x_train.shape)
    print("x_test.shape= ", x_test.shape)
    print("train_label.shape= ", train_label.shape)
    print("test_label.shape= ", test_label.shape)
    print("data.shape= ", data.shape)
    print("labels.shape= ", labels.shape)
    
    def check_missing_values(df):
        if df.isnull().values.any():
            print("The DataFrame has missing values.")
        else:
            print("The DataFrame doesn't have any missing values.")

    # Check for missing values in train, test, data, and labels
    check_missing_values(train)
    check_missing_values(test)
    check_missing_values(data)
    check_missing_values(labels)
    
    # data = data.fillna("")
    
    longest = pd.DataFrame([max(data, key = len)]).iloc[:,0]
    print("longest sequence= ", longest)
    data = data.append(longest).reset_index(drop=True)
    
    per_residue = True 
    data = embedding_out(data,data,per_residue)
    data = np.delete(data,-1,axis=0)
    print("train_encoding.shape",data.shape)

    x_train_encoding = data[:len(x_train),:,:]
    x_test_encoding = data[len(x_train):,:,:]
    
    print("train_encoding.shape",x_train_encoding.shape)
    print("test_encoding.shape",x_test_encoding.shape)

    # ----------------- For S/T dataset ---------------------
    # np.save('./embedding/ST_train_embedding.npy',x_train_encoding)
    # np.save('./embedding/ST_test_embedding.npy',x_test_encoding)    
    # ------------------------------------------------------

    # ----------------- For Y dataset ----------------------
    np.save('./embedding/Y_train_embedding.npy',x_train_encoding)
    np.save('./embedding/Y_test_embedding.npy',x_test_encoding)    
    # ------------------------------------------------------