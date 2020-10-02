#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 17:37:54 2020

@author: lenovo
"""
import numpy as np
import matplotlib.pyplot as plt
import os

path_to_results = '/home/abel/Desktop/VoxNet_Specie_Sauvegarde/Metrics_specie/'
train_loss_name = 'train_loss2.npy'
metrics_name = 'train_metrics2.npy'

t = []
train_loss = list(np.load(os.path.join(path_to_results, train_loss_name)))
for i in range(len(train_loss)) :
    for j in range(len(train_loss[0])) :
        t.append(train_loss[i][j])
metrics = list(np.load(os.path.join(path_to_results, metrics_name)))

if __name__ == '__main__' :
    print(t)
    print(metrics)