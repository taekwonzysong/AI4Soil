# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:37:14 2024

@author: Zheyu Jiang
"""

import pandas as pd
import torch

def load_data1(file_path):
    datas = pd.read_csv(file_path, header=None)
    datas.columns = ["0", "1"]
    nw1 = datas["0"].values * 1e-11
    pw = datas["1"].values * 1e-1
    x = torch.FloatTensor(nw1).unsqueeze(1)
    y = torch.FloatTensor(pw).unsqueeze(1)
    return x, y

def load_data2(file_path):
    # Load the CSV data using pandas
    data = pd.read_csv(file_path)
    x = torch.tensor(data['x'].values, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(data['y'].values, dtype=torch.float32).view(-1, 1)
    return x,y
    