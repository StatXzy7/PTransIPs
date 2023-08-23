# PTransIPs

PTransIPs: Identification of SARS-CoV-2 phosphorylation sites based on protein pretrained model embedding and transformer [[Paper]](https://arxiv.org/abs/2308.05115)

**The details of code tutorial will be updated soon in August, 2023.**



## Approach

![flowchart](flowchart.png)

## Usage



### Step1: Generate two pretrained embedding

(**Optional**, there are complete embeddings for Y sites in the data folder.)



#### Input1: Original sequence

fasta/csv sequence file, name 

#### Input2: Sequence pretrained embedding

To generate the sequence pretrained embedding, use `pretrained_embedding_generate.py` to do the following steps:

```bash
$ !pip install torch transformers sentencepiece h5py
$ python model_train_test/pretrained_embedding_generate.py
```

Details in this part please refer to **[ProtTrans](https://github.com/agemagician/ProtTrans)**.

#### Input3: Structure pretrained embedding

First git clone the `EMBER2` project, and then move the file `pretrained_embedding_generate.py` into the `EMBER2` folder to use the model for generating the predicted structures for the current sequence.

To generate the sequence pretrained embedding, use `pretrained_embedding_generate.py` to do the following steps:

```bash
$ git clone https://github.com/kWeissenow/EMBER2.git
$ cp model_train_test/structure_embedding_generate.py EMBER2/
$ python EMBER2/structure_embedding_generate.py -i "data/Y-train.fa" -o "EMBER2/output"
$ python EMBER2/structure_embedding_generate.py -i "data/Y-test.fa" -o "EMBER2/output"
```

Details in this part please refer to **[EMBER2](https://github.com/kWeissenow/EMBER2)**.



### Step2: Training Model

Use `train.py` to train the PTransIPs model in `PTransIPs_model.py`.

```bash
$ python model_train_test/train.py
```



### Step3: Evaluate the model performance on independent testset
Use `model_performance _evaluate.py.py` to evaluate the model performance on independent testset.

```bash
$ python model_train_test/model_performance _evaluate.py.py
```



### Step4: Other analysis
Use `umap.py` to generate umap visualization.

Use `Generate_tfseq.py` files to generate sequence for two sample logos. 

