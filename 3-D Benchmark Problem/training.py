# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:38:42 2024

@author: Zheyu Jiang
"""

import torch

def train_model(model, optimizer, x, y, epoch):
    loss_list = []
    for _ in range(epoch):
        preds = model(x)
        loss = torch.nn.functional.mse_loss(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    return loss_list
