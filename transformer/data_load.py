# -*- coding: utf-8 -*-
#/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

truncate_size = 200

import tensorflow as tf
from utils import calc_num_batches

def read_data(data_path=None):
    train_data=[]
    label_train_target=[]
    mask=[];

    #iterate the records
    train_content=open(data_path, 'r')
    train_records=[line.rstrip('\n') for line in train_content]
    c_uid=-1
    list_r=[]
    target_list=[]
    mask_list=[]
    counter = 0
    for record_index in range(len(train_records)-1):
        columns=train_records[record_index].split(",")
        if(record_index==0):
            c_uid=columns[0]
        if(columns[0]!=c_uid or counter>=truncate_size):
            train_data.append(list_r)
            label_train_target.append(target_list)
            mask.append(mask_list)
            list_r=[]
            target_list=[]
            mask_list=[];
            c_uid=columns[0]
            counter = 0
        list_r.append(columns[1])
        list_r.append(columns[2])
        list_r.append(columns[3])
        list_r.append(columns[4])
        counter += 1
        if(len(columns) >= 6):
            target_list.append(columns[5])
            mask_list.append(1)
        else:
            target_list.append(0)
            mask_list.append(0)
    return train_data, label_train_target, mask

def TD_raw_data(data_path=None, data_file=None):
	#set the data path
	train_path=data_path
	
	train_data, label_train_target, train_mask = read_data(data_path=train_path)
								
	return train_data, label_train_target, train_mask

def data_padding(data, feature_dim):
    num_samples=len(data)
    lengths=[int(len(s)) for s in data]
    max_length=max(lengths)
    padding_dataset=np.full((num_samples, max_length), 0, dtype=np.int)
    for idx, seq in enumerate(data):
        padding_dataset[idx, :len(seq)]=seq
    for idx in range(len(lengths)):
        lengths[idx]=lengths[idx]//feature_dim
    return padding_dataset, lengths

#slicing
def data_slicing(data, feature_dim):
    num_samples=len(data)
    lengths=[int(len(s)) for s in data]
    min_length=min(lengths)
    slicing_dataset=np.full((num_samples, min_length), 0, dtype=np.int)
    for idx, seq in enumerate(data):
        slicing_dataset[idx, :]=seq[:min_length]
    for idx in range(len(lengths)):
        lengths[idx]=lengths[idx]//feature_dim
    return slicing_dataset, lengths
    

def generator_fn(path):

    train_data, label_train_target, mask = TD_raw_data(path)
    
    for i in range(len(train_data)):
        input_hour=[]
        input_minutes=[]
        input_lat=[]
        input_lon=[]
        for j in range(len(train_data[i])):
            t=j%4
            if t==0:
                input_hour.append(train_data[i][j])
            if t==1:
                input_minutes.append(train_data[i][j])
            if t==2:
                input_lat.append(train_data[i][j])
            if t==3:
                input_lon.append(train_data[i][j])
        label_train_target[i].insert(0,0)
        
        x_seqlen, y_seqlen = len(train_data[i]), len(label_train_target[i])
        yield (input_hour, input_minutes, input_lat, input_lon, x_seqlen), (label_train_target[i][:-1], label_train_target[i][1:], y_seqlen), (mask[i])


def input_fn(path, batch_size, shuffle=False):
    
    shapes = (([None], [None], [None], [None], ()),
              ([None], [None], ()),([None]))
    types = ((tf.int32, tf.int32, tf.int32, tf.int32, tf.int32),
             (tf.int32, tf.int32, tf.int32),(tf.int32))
    paddings = ((0, 0, 0, 0, 0),
                (0, 0, 0),(0))
    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(path,))  # <- arguments for generator_fn. converted to np string arrays
    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)
    return dataset

def get_batch(path, batch_size, shuffle=False):
    batches = input_fn(path, batch_size, shuffle=shuffle)
    train_data, label_train_target, mask = TD_raw_data(path)
    num_batches = calc_num_batches(len(train_data), batch_size)
    print(len(train_data))
    print(num_batches)
    return batches, num_batches, len(train_data)
