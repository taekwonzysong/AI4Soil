# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:36:37 2024

@author: Zheyu Jiang
"""

import torch
import torch.nn as nn
class MLP1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=torch.nn.Linear(1,256)
        self.layer2=torch.nn.Linear(256,256)
        self.layer3=torch.nn.Linear(256,256)
        self.layer4=torch.nn.Linear(256,256)
        self.layer5=torch.nn.Linear(256,1)
        #self.layer6=torch.nn.Linear(256,256)
        #self.layer7=torch.nn.Linear(256,1)
    def forward(self,x):
        x=self.layer1(x)
        x=torch.nn.functional.leaky_relu(x)
        x=self.layer2(x)
        x=torch.nn.functional.leaky_relu(x)
        x=self.layer3(x)
        x=torch.nn.functional.leaky_relu(x)
        x=self.layer4(x)
        x=torch.nn.functional.leaky_relu(x)
        x=self.layer5(x)
        #x=torch.nn.functional.leaky_relu(x)
        #x=self.layer6(x)
        #x=torch.nn.functional.leaky_relu(x)
        #x=self.layer7(x)
        return x

class MLP2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=torch.nn.Linear(1,256)
        self.layer2=torch.nn.Linear(256,256)
        self.layer3=torch.nn.Linear(256,256)
        self.layer4=torch.nn.Linear(256,256)
        self.layer5=torch.nn.Linear(256,1)
        
    def forward(self,x):
        x=self.layer1(x)
        x=torch.nn.functional.leaky_relu(x)
        x=self.layer2(x)
        x=torch.nn.functional.leaky_relu(x)
        x=self.layer3(x)
        x=torch.nn.functional.leaky_relu(x)
        x=self.layer4(x)
        x=torch.nn.functional.leaky_relu(x)
        x=self.layer5(x)
        return x

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear = nn.Linear(1, 1, bias=False)  # No bias since y = k * x

    def forward(self, x):
        return self.linear(x)