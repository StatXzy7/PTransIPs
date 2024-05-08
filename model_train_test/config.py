# ---encoding:utf-8---

import argparse
import torch


def get_train_config():
    parse = argparse.ArgumentParser(description='iDNA_ABT train model')

    # preoject setting
    parse.add_argument('--Y', action='store_true', help='Select the Y dataset')
    parse.add_argument('--ST', action='store_true', help='Select the ST dataset')
    parse.add_argument('--device', type=int, default=0, choices=list(range(torch.cuda.device_count())),help='ordinal number of the GPU to use for computation')
    parse.add_argument('-learn-name', type=str, default='BERT_validation', help='learn name')
    parse.add_argument('-save-best', type=bool, default=True, help='if save parameters of the current best model ')
    parse.add_argument('-threshold', type=float, default=0.8, help='save threshold')
    parse.add_argument('-vocab_size', type=int, default=28, help='vocab_size')
    # model parameters
    parse.add_argument('-max-len', type=int, default=86, help='max length of input sequences')
    parse.add_argument('-num-layer', type=int, default=6, help='number of encoder blocks')  # 3
    parse.add_argument('-num-head', type=int, default=8, help='number of head in multi-head attention')  # 8
    # parse.add_argument('-dim-embedding', type=int, default=128, help='residue embedding dimension')  # 64
    parse.add_argument('-dim-embedding', type=int, default=1024, help='residue embedding dimension')  # 64
    parse.add_argument('-dim-feedforward', type=int, default=32, help='hidden layer dimension in feedforward layer')
    parse.add_argument('-dim-k', type=int, default=32, help='embedding dimension of vector k or q')
    parse.add_argument('-dim-v', type=int, default=32, help='embedding dimension of vector v')
    parse.add_argument('-num-embedding', type=int, default=1, help='number of sense in multi-sense')
    parse.add_argument('-k-mer', type=int, default=1, help='number of k(-mer) in multi-sccaled')
    parse.add_argument('-embed-atten-size', type=int, default=8, help='size of soft attention')
    parse.add_argument('-dropout', type=float, default=0.4, help='dropout rate')

    # training parameters
    parse.add_argument('-lr', type=float, default=1e-3, help='learning rate') #learning rate
    parse.add_argument('-reg', type=float, default=0.0025, help='weight lambda of regularization')
    parse.add_argument('-batch-size', type=int, default=128, help='number of samples in a batch') #batch
    parse.add_argument('-epoch', type=int, default=200  , help='number of iteration')  # epoch
    parse.add_argument('-k-fold', type=int, default=-1, help='k in cross validation,-1 represents train-test approach')
    parse.add_argument('-num-class', type=int, default=2, help='number of classes')
    # parse.add_argument('-cuda', type=bool, default=True, help='if use cuda')
    parse.add_argument('-cuda', type=bool, default=True, help='if not use cuda')
    # parse.add_argument('-device', type=int, default='0', help='device id')
    parse.add_argument('-interval-log', type=int, default=5,
                       help='how many batches have gone through to record the training performance')
    parse.add_argument('-interval-valid', type=int, default=1,
                       help='how many epoches have gone through to record the validation performance')  # 20
    parse.add_argument('-interval-test', type=int, default=1,
                       help='how many epoches have gone through to record the test performance')
    parse.add_argument('-alpha', type=float, default=0.10, help='information entropy')
    
    parse.add_argument('-task', type=str, default='train', help='information entropy')
    config = parse.parse_args()
    return config
