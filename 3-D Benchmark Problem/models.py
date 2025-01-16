# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:36:37 2024

@author: Zheyu Jiang
"""

import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

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
        x=torch.nn.functional.relu(x)
        x=self.layer2(x)
        x=torch.nn.functional.relu(x)
        x=self.layer3(x)
        x=torch.nn.functional.relu(x)
        x=self.layer4(x)
        x=torch.nn.functional.relu(x)
        x=self.layer5(x)
        #x=torch.nn.functional.relu(x)
        #x=self.layer6(x)
        #x=torch.nn.functional.relu(x)
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
        x=torch.nn.functional.relu(x)
        x=self.layer2(x)
        x=torch.nn.functional.relu(x)
        x=self.layer3(x)
        x=torch.nn.functional.relu(x)
        x=self.layer4(x)
        x=torch.nn.functional.relu(x)
        x=self.layer5(x)
        return x

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear = nn.Linear(1, 1, bias=False)  

    def forward(self, x):
        return self.linear(x)

class myLayer(tf.keras.layers.Layer):
    def __init__(self, a, **kwargs):
        super(myLayer, self).__init__(**kwargs)
        if len(a.shape) == 3:
            a = a[..., np.newaxis]
        self.a = tf.convert_to_tensor(a, dtype=tf.float32)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        out = tf.expand_dims(self.a, axis=0)
        out = tf.repeat(out, repeats=batch_size, axis=0)
        out = tf.squeeze(out, axis=-1)
        return out

    def get_config(self):
        config = super(myLayer, self).get_config()
        config.update({
            "a": self.a.numpy().tolist()
        })
        return config

    @classmethod
    def from_config(cls, config):
        a = tf.convert_to_tensor(config.pop("a"), dtype=tf.float32)
        return cls(a, **config)

def MLP2D(input_data):
    inputs = tf.keras.Input(shape=(21, 21, 21, 1))

    x = layers.Conv3D(
        filters=8, 
        kernel_size=1, 
        activation='relu',
        name='conv3d_1'
    )(inputs)

    x = layers.Conv3D(
        filters=8, 
        kernel_size=1, 
        activation='relu',
        name='conv3d_2'
    )(x)

    outputs = myLayer(input_data, name='output_layer')(x)
    model = Model(inputs=inputs, outputs=outputs, name="mlp_2d_model")
    return model