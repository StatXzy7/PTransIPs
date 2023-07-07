# PTransIPs
PTransIPs: Identification of SARS-CoV-2 phosphorylation sites based on protein pretrained model embedding and transformer

## Step1: Generate two pretrained embedding
fasta file: Original sequence as input1;

Use `pretrained_embedding_generate.py` to generate sequence pretrained embedding as input2;

Use `structure_embedding_generate.py` to generate structure pretrained embedding as input3.

## Step2: Training Model
Use `bert_train_pretrain_final.py.py` to train the PTransIPs model.

## Step3: Evaluate the model performance on independent testset
Use `model_performance _evaluate.py.py` to evaluate the model performance on independent testset.

## Step4: Other analysis
Use `umap.py` to generate umap visualization.

Use `Generate_tfseq.py` files to generate sequence for two sample logos. 