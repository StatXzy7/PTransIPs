# PTransIPs
![](https://img.shields.io/badge/PRs-Welcome-blue)
![](https://img.shields.io/github/last-commit/StatXzy7/PTransIPs?color=green)

PTransIPs: Identification of SARS-CoV-2 phosphorylation sites based on protein pretrained model embedding and transformer [[Paper]](https://arxiv.org/abs/2308.05115)



## Architecture

![flowchart](flowchart.png)

## Table of Contents

- [1. Steup](#1-setup)
- [2. Generate two pretrained embedding](#2-generate-two-pretrained-embedding)
    * [Input1: Sequence](#input1-sequence)
    * [Input2: Sequence pretrained embedding](#input2-sequence-pretrained-embedding)
    * [Input3: Structure pretrained embedding](#input3-structure-pretrained-embedding)
- [3. Training PTransIPs Model](#3-training-ptransips-model)
- [4. Evaluate the model performance on independent testset](#4-evaluate-the-model-performance-on-independent-testset)
- [5. Some Visualization Analysis](#5-some-visualization-analysis)

## 1. Setup

#### 🔧Pip Installation

**Note: We recommend use Python 3.9 for PTransIPs, and use conda to manage your environments!**

To get started, simply install conda and run:

```shell
git clone https://github.com/StatXzy7/PTransIPs.git
conda create --name PTransIPs python==3.9
...
pip install -r requirements.txt
```

## 2. Generate two pretrained embedding

(**For ones that wish to skip this step:** We have already uploaded complete embeddings for Y sites in the data folder `./embedding/`. For S/T sites, you may download complete embeddings from [**All PTransIPs pretrained embeddings**](https://1drv.ms/f/s!AqzWnkSOWHpvhxMUDCjM9KFpz50O?e=N23jEn) and place them under the directory`./embedding/`)


### Input1: Sequence

The orginal fasta/csv sequence file already exists in `./data/`.

### Input2: Sequence pretrained embedding

To generate sequence pretrained embedding, run `./src/pretrained_embedding_generate.py` directly:

```bash
python src/pretrained_embedding_generate.py
```

**The code is set to generate embeddings for Y sites as default, if you attempt to do that for S/T sites, you should run the code after commenting Y sites' part and uncommenting S/T sites' part!**

You may also refer to **[ProtTrans](https://github.com/agemagician/ProtTrans)** for detailed explanations.

### Input3: Structure pretrained embedding

To generate structure embeddding, firstly, git clone the `EMBER2` project. After moving the file `./src/pretrained_embedding_generate.py` into the `EMBER2` folder, you may run the codes: 

```bash
git clone https://github.com/kWeissenow/EMBER2.git
cp src/structure_embedding_generate.py EMBER2/
python EMBER2/structure_embedding_generate.py -i "data/Y-train.fa" -o "EMBER2/output"
python EMBER2/structure_embedding_generate.py -i "data/Y-test.fa" -o "EMBER2/output"
```
**Here, `structure_embedding_generate.py` is set to generate embeddings for Y sites as default, if you attempt to do that for S/T sites, you may run as follows after modify the codes by commenting Y sites' part and uncommenting S/T sites' part!**

```bash
python EMBER2/structure_embedding_generate.py -i "data/ST-train.fa" -o "EMBER2/output"
python EMBER2/structure_embedding_generate.py -i "data/ST-test.fa" -o "EMBER2/output"
```

You may also refer to **[EMBER2](https://github.com/kWeissenow/EMBER2)** for detailed explanations.



## 3. Training PTransIPs Model

(**For ones that wish to skip this step:** you may [**Download the PTransIPs model**](https://1drv.ms/f/s!AqzWnkSOWHpvhxMUDCjM9KFpz50O?e=N23jEn) directly. Remember to place them under `.\model\Y_train` or `.\model\ST_train` so that you can proceed to the evaluation step directly.)

Run `./src/train.py` to train the PTransIPs model in `./src/PTransIPs_model.py`.

Important parameters are:
1. ``--Y``: To specify that we train the model on Y sites.
2. ``--ST``: To specify that we train the model on ST sites.
3. ``--device``: To specify which GPU to train the model on. (input an integer to specify, default is ``cuda:0``)

Example: Train PTransIPs on ST sites with default GPU:

```bash
python src/train.py --ST
```



## 4. Evaluate the model performance on independent testset

Run `./src/model_performance_evaluate.py` to evaluate the model performance on independent testset.

Important parameters are:
1. ``--Y``: To specify that we evalute the model trained on Y sites.
2. ``--ST``: To specify that we evaluate the model trained on ST sites.
3. ``--path``: To specify the path of model we evaluate, if you trained as default code, you should specify ``./model/Y_train`` for Y sites and ``./model/ST_train`` for ST sites.(but this part CAN't be empty!)

Example: Evaluate PTransIPs model trained on Y sites with default path:

```bash
python src/model_performance_evaluate.py \
        --Y \
        --path ./model/Y_train
```

Files `path/PTransIPs_test_prob.npy` and `path/PTransIPs_text_result.txt` will be created, representing the prediction probability and performance of PTransIPs, respectively. (where `path/` depends on which sites you choose`)



## 5. Some Visualization Analysis

**You can see the results directly in the files uploaded, in the directory `figures/umap_pdf`**.

Run `./src/umap_test.py` to generate umap visualization figures. Remember to modify the path of the model to the one that you want to visualize.

```bash
python src/umap_test_Y.py
python src/umap_test_ST.py
```

Run `./src/Generate_tfseq.py` files to generate sequence for Two Sample Logo analysis. Remember to modify the path of the model to the one that you want to visualize.

```bash
python src/Generate_tfseq_Y.py
python src/Generate_tfseq_ST.py
```

## Citation
Please feel free to email us at `ziyangxu0205@gmail.com` or `haitian.zhong@cripac.ia.ac.cn`. If you find this work useful in your own research, please consider citing our work. 
```bibtex
@ARTICLE{xu2024ptransips,
  author={Xu, Ziyang and Zhong, Haitian and He, Bingrui and Wang, Xueying and Lu, Tianchi},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={PTransIPs: Identification of Phosphorylation Sites Enhanced by Protein PLM Embeddings}, 
  year={2024},
  volume={},
  number={},
  pages={1-10},
  keywords={Proteins;Protein engineering;Amino acids;Training;Biological system modeling;Data models;Vectors;Phosphorylation sites;protein pre-trained language model;CNN;Transformer},
  doi={10.1109/JBHI.2024.3377362}}
```