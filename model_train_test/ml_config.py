# -*- coding: utf-8 -*-
"""
@author: ZiyangXu
"""

import numpy as  np
import pandas as pd
import argparse

def Logistic_config():
    parse = argparse.ArgumentParser(description='Logistic_Regression')
    parse.add_argument('-fit_intercept',type=bool, default=True,help='fit_intercept')
    parse.add_argument('-max_iter',type=int,default=200,help='max_iter')
    parse.add_argument('-penalty',type=str,default='none',help='penalty')
    parse.add_argument('-tol',type=float,default=10e-4,help='tol')
    config = parse.parse_args()
    return config

def Linear_SVM_config():
    parse = argparse.ArgumentParser(description='Linear_SVM')
    parse.add_argument('-')
    config = parse.parse_args()
    return config