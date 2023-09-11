# PTransIPs

PTransIPs: Identification of SARS-CoV-2 phosphorylation sites based on protein pretrained model embedding and transformer [[Paper]](https://arxiv.org/abs/2308.05115)



## Approach

![flowchart](flowchart.png)

## Usage



### Step1: Generate two pretrained embedding

(**Optional**, there are complete embeddings for Y sites in the data folder. Or you can download the complete embeddings below)
[**Download all the PTransIPs pretrained embeddings**](https://1drv.ms/f/s!AqzWnkSOWHpvhxMUDCjM9KFpz50O?e=N23jEn)


#### Input1: Sequence

fasta/csv sequence file

#### Input2: Sequence pretrained embedding

To generate the sequence pretrained embedding, use `pretrained_embedding_generate.py` to do the following steps:

```bash
$ !pip install torch transformers sentencepiece h5py
$ python model_train_test/pretrained_embedding_generate.py
```

For detailed guide in this part, please refer to **[ProtTrans](https://github.com/agemagician/ProtTrans)**.

#### Input3: Structure pretrained embedding

First git clone the `EMBER2` project, and then move the file `pretrained_embedding_generate.py` into the `EMBER2` folder to use the model for generating the predicted structures for the current sequence.

To generate the sequence pretrained embedding, use `pretrained_embedding_generate.py` to do the following steps:

```bash
$ git clone https://github.com/kWeissenow/EMBER2.git
$ cp model_train_test/structure_embedding_generate.py EMBER2/
$ python EMBER2/structure_embedding_generate.py -i "data/Y-train.fa" -o "EMBER2/output"
$ python EMBER2/structure_embedding_generate.py -i "data/Y-test.fa" -o "EMBER2/output"
```

For detailed guide in this part, please refer to **[EMBER2](https://github.com/kWeissenow/EMBER2)**.



### Step2: Training PTransIPs Model

**You can proceed directly to this step**, as the requisite pretrained embeddings of dataset (Y sites) have been uploaded to GitHub.

Run `train.py` to train the PTransIPs model in `PTransIPs_model.py`.

```bash
$ python model_train_test/train.py
```



### Step3: Evaluate the model performance on independent testset

[**Download the PTransIPs model**](https://1drv.ms/f/s!AqzWnkSOWHpvhxMUDCjM9KFpz50O?e=N23jEn)

**You can proceed directly to this step**, if you have downloaded the models and put it into the `PTransIPs` folder

Run `model_performance _evaluate.py` to evaluate the model performance on independent testset.

```bash
$ python model_train_test/model_performance_evaluate.py
```

This function will create files `PTransIPs_test_prob.npy` and `PTransIPs_text_result.txt`, represent the prediction probability and performance of PTransIPs, respectively.



### Step4: Other Visual analysis

**You can proceed directly to this step**, if you have downloaded the models and put it into the `PTransIPs` folder

**You can also see the results directly in the GitHub**.

Run `umap.py` to generate umap visualization figures.

```bash
$ python model_train_test/umap_test_Y.py
```

Run `Generate_tfseq.py` files to generate sequence for Two Sample Logo analysis. 

```bash
$ python model_train_test/Generate_tfseq_Y.py
```



