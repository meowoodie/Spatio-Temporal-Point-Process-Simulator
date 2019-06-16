#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo for testing basic version of imitpp on synthetic dataset
"""

import sys
import arrow
import random
import numpy as np
import tensorflow as tf

def split_crime_by_days(datapath, days, morethan=10):
    with open(datapath, "r") as f:
        data = [ line.strip("\n").split("\t") for line in f ]
        data = [ [ 
                arrow.get(d[0][:-4], "YYYY-MM-DD HH:mm:ss").timestamp, 
                float(d[1])/np.power(10, len(d[1])-3),
                -1 * float(d[2])/np.power(10, len(d[2])-3)] 
            for d in data 
            if d[0].strip() != "" and d[1].strip() != "" and d[2].strip() != ""]
        data = [ d for d in data if d[1] >= 32 and d[1] < 34 and d[2] >= -85 and d[2] < -83]
        data = np.array(data)

    # reorder the dataset by timestamps
    order = data[:, 0].argsort()
    data  = data[order]
    # start time
    stack_list = []
    last_i     = 0
    max_len    = 0
    for i in range(len(data)):
        if data[i, 0] - data[last_i, 0] > days * 24 * 3600:
            seq = data[last_i:i, :]
            seq[:, 0] -= seq[0, 0]
            max_len = len(seq) if len(seq) > max_len else max_len
            if len(seq) >= morethan:
                stack_list.append(seq)
            last_i = i
    new_data = np.zeros((len(stack_list), max_len, 3))
    for j in range(len(stack_list)):
        new_data[j, :len(stack_list[j]), :] = stack_list[j]
    return new_data

def split_earthquake_by_days(datapath, days, morethan=10):
    with open(datapath, "r") as f:
        data = [ line.strip("\n").split(",") for line in f ]
        data = [ [ 
                arrow.get(d[0][:-3], "YYYY/MM/DD HH:mm:ss").timestamp, 
                float(d[1]),
                float(d[2])] 
            for d in data 
            if d[0].strip() != "" and d[1].strip() != "" and d[2].strip() != ""]
        data = np.array(data)

    # reorder the dataset by timestamps
    order = data[:, 0].argsort()
    data  = data[order]
    # start time
    stack_list = []
    last_i     = 0
    max_len    = 0
    for i in range(len(data)):
        if data[i, 0] - data[last_i, 0] > days * 24 * 3600:
            seq = data[last_i:i, :]
            seq[:, 0] -= seq[0, 0]
            max_len = len(seq) if len(seq) > max_len else max_len
            if len(seq) >= morethan:
                stack_list.append(seq)
            last_i = i
    new_data = np.zeros((len(stack_list), max_len, 3))
    for j in range(len(stack_list)):
        new_data[j, :len(stack_list[j]), :] = stack_list[j]
    return new_data

if __name__ == "__main__":

    days  = 30
    # expert_seq = split_crime_by_days("data/apd.robbery.txt", days, morethan=10)
    expert_seq = split_earthquake_by_days("data/northcal.earthquake.txt", days, morethan=10)

    T         = expert_seq[:, :, 0].max() # days * 24 * 3600
    n_seq     = expert_seq.shape[0]
    step_size = expert_seq.shape[1]
    x_nonzero = expert_seq[:, :, 1][np.nonzero(expert_seq[:, :, 1])]
    y_nonzero = expert_seq[:, :, 2][np.nonzero(expert_seq[:, :, 2])]
    xlim      = [ x_nonzero.min(), x_nonzero.max() ]
    ylim      = [ y_nonzero.min(), y_nonzero.max() ]

    np.save('data/northcal.earthquake.permonth.npy', expert_seq)

    print(expert_seq.shape)
    print("T", T)
    print("n_seq", n_seq)
    print("step_size", step_size)
    print("xlim", xlim)
    print("ylim", ylim)
    print(expert_seq[1, :, :])