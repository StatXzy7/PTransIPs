from transformers import T5EncoderModel, T5Tokenizer
import torch
import h5py
import time
import numpy as np
import pandas as pd

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
    # test_path = "./protT5/test.fasta"
    # train = pd.read_csv(r"C:\Users\tianchilu4\Desktop\project1.1\data\neoantigen_train.csv",header=0)
    # train = pd.read_csv('../data/train_val2.csv',header=0)
    # train = pd.read_csv('../data/1DATA.csv',header=0)
    
    # train = pd.read_csv('program 1.1/data/DeepIPS_Train_data.csv',header=0)
    # # train = pd.read_csv('../data/TCR_state_train_data.csv',header=0)
    # # train = pd.read_csv('../data/TCR_positive.csv',header=0)
    # # x_train_positive = train[train.apply(lambda x: x.iloc[1]==1,axis=1)].iloc[:,0]
    # # x_train_negative = train[train.apply(lambda x : x.iloc[1]==0,axis=1)].iloc[:,0]
    
    # x_train = train.iloc[:,1]
    # longest = pd.DataFrame([max(x_train, key = len)]).iloc[:,0]
    # x_train = x_train.append(longest).reset_index(drop=True)
    
    # # x_train = x_train.drop_duplicates(keep='first')reset_index()
    # # x_train = x_train['Epitope']
    
    # per_residue = True 
    # embedding = embedding_out(x_train,x_train,per_residue)
    # embedding = np.delete(embedding,-1,axis=0)
    
    # positive_embedding = embedding_out(x_train_positive,x_train_positive,per_residue)
    # negative_embedding = embedding_out(x_train_negative,x_train_negative,per_residue)
    # np.save('IEDB_HLA_embedding_positive.npy',positive_embedding)
    # np.save('IEDB_HLA_embedding_negative.npy',negative_embedding)
    # np.save('CEDAR_IEDB_embedding.npy',embedding)
    
    # np.save('embedding.npy',embedding)
    # embedding = embeddings(x_train,x_train,per_residue)
    

    train = pd.read_csv("/root/myDNAPredict/program 1.1/data/DeepIPS_Train_data.csv",header=0)
    test = pd.read_csv("/root/myDNAPredict/program 1.1/data/DeepIPS_Test_data.csv",header=0)
    x_train = train.iloc[:,1]
    x_test = test.iloc[:,1]
    train_label = train.iloc[:,0]
    test_label = test.iloc[:,0]
    
    longest_train = pd.DataFrame([max(x_train, key = len)]).iloc[:,0]
    x_train = x_train.append(longest_train).reset_index(drop=True)
    longest_test = pd.DataFrame([max(x_test, key = len)]).iloc[:,0]
    x_test = x_test.append(longest_test).reset_index(drop=True)
    
    per_residue = True 
    # embedding = embedding_out(x_train,x_train,per_residue)
    # embedding = np.delete(embedding,-1,axis=0)
    x_train_encoding = embedding_out(x_train,x_train,per_residue)
    x_train_encoding = np.delete(x_train_encoding,-1,axis=0)
    x_test_encoding = embedding_out(x_test,x_test,per_residue)
    x_test_encoding = np.delete(x_test_encoding,-1,axis=0)

    np.save('/root/autodl-tmp/x_train_embedding.npy',x_train_encoding)
    np.save('/root/autodl-tmp/x_test_embedding.npy',x_test_encoding)