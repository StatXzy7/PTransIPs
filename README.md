# PTransIPs

PTransIPs: Identification of SARS-CoV-2 phosphorylation sites based on protein pretrained model embedding and transformer [[Paper]](https://arxiv.org/abs/2308.05115)



## Architecture

![flowchart](flowchart.png)

## Table of Contents

- [Step1: Generate two pretrained embedding](#step1-generate-two-pretrained-embedding)
    * [Input1: Sequence](#input1-sequence)
    * [Input2: Sequence pretrained embedding](#input2-sequence-pretrained-embedding)
    * [Input3: Structure pretrained embedding](#input3-structure-pretrained-embedding)
- [Step2: Training PTransIPs Model](#step2-training-ptransips-model)
- [Step3: Evaluate the model performance on independent testset](#step3-evaluate-the-model-performance-on-independent-testset)
- [Step4: Other Visual analysis](#step4-other-visual-analysis)

## Step1: Generate two pretrained embedding

(**For ones that wish to skip this step:** We have already uploaded complete embeddings for Y sites in the data folder `./embedding/`. For S/T sites, you may download complete embeddings from [**All PTransIPs pretrained embeddings**](https://1drv.ms/f/s!AqzWnkSOWHpvhxMUDCjM9KFpz50O?e=N23jEn) and place them under the directory`./embedding/`)


### Input1: Sequence

The orginal fasta/csv sequence file already exists in `./data/`.

### Input2: Sequence pretrained embedding

To generate sequence pretrained embedding, run `./model_train_test/pretrained_embedding_generate.py` directly:

```bash
python model_train_test/pretrained_embedding_generate.py
```

**The code is set to generate embeddings for Y sites as default, if you attempt to do that for S/T sites, you should run the code after commenting Y sites' part and uncommenting S/T sites' part!**

You may also refer to **[ProtTrans](https://github.com/agemagician/ProtTrans)** for detailed explanations.

### Input3: Structure pretrained embedding

To generate structure embeddding, firstly, git clone the `EMBER2` project. After moving the file `./model_train_test/pretrained_embedding_generate.py` into the `EMBER2` folder, you may run the codes: 

```bash
git clone https://github.com/kWeissenow/EMBER2.git
cp model_train_test/structure_embedding_generate.py EMBER2/
python EMBER2/structure_embedding_generate.py -i "data/Y-train.fa" -o "EMBER2/output"
python EMBER2/structure_embedding_generate.py -i "data/Y-test.fa" -o "EMBER2/output"
```
**Here, `structure_embedding_generate.py` is set to generate embeddings for Y sites as default, if you attempt to do that for S/T sites, you may run as follows after modify the codes by commenting Y sites' part and uncommenting S/T sites' part!**

```bash
python EMBER2/structure_embedding_generate.py -i "data/ST-train.fa" -o "EMBER2/output"
python EMBER2/structure_embedding_generate.py -i "data/ST-test.fa" -o "EMBER2/output"
```

You may also refer to **[EMBER2](https://github.com/kWeissenow/EMBER2)** for detailed explanations.



## Step2: Training PTransIPs Model

(**For ones that wish to skip this step:** you may [**Download the PTransIPs model**](https://1drv.ms/f/s!AqzWnkSOWHpvhxMUDCjM9KFpz50O?e=N23jEn) directly. Remember to place them under `.\model\Y_train` or `.\model\ST_train` so that you can proceed to the evaluation step directly.)

Run `./model_train_test/train.py` to train the PTransIPs model in `./model_train_test/PTransIPs_model.py`:

(**Note that `train.py` is set to train Y sites as default, if you attempt to train S/T sites, you'll have to modify the codes by commenting Y sites' part and uncommenting S/T sites' part!**)

```bash
python model_train_test/train.py
```



## Step3: Evaluate the model performance on independent testset

Run `./model_train_test/model_performance _evaluate.py` to evaluate the model performance on independent testset.

(**Still, `model_performance _evaluate.py` is set to evaluate the model trained on Y sites as default, if you attempt to evaluatet that of S/T sites, you can run as follows after modify the codes by commenting Y sites' part and uncommenting S/T sites' part!**)

```bash
python model_train_test/model_performance_evaluate.py
```

Files `path/PTransIPs_test_prob.npy` and `path/PTransIPs_text_result.txt` will be created, representing the prediction probability and performance of PTransIPs, respectively. (where `path/` depends on which sites you choose`)



## Step4: Other Visual analysis

**You can see the results directly in the files uploaded, in the directory `figures/umap_pdf`**.

Run `./model_train_test/umap_test.py` to generate umap visualization figures. Remember to modify the path of the model to the one that you want to visualize.

```bash
python model_train_test/umap_test_Y.py
python model_train_test/umap_test_ST.py
```

Run `./model_train_test/Generate_tfseq.py` files to generate sequence for Two Sample Logo analysis. Remember to modify the path of the model to the one that you want to visualize.

```bash
python model_train_test/Generate_tfseq_Y.py
python model_train_test/Generate_tfseq_ST.py
```



