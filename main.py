# import pickle # to load pickle data
import numpy as np  # for matrix multiplication
from pickle_handler import PickleHandler
from keras.layers import (
    LSTM,
    Activation,
    Dropout,
    Dense,
    Input,
)  # layers to make models
from keras.models import Model, Sequential  # model class to make a model
from keras.callbacks import TensorBoard  # for tensorboard ops

import tensorflow as tf  # main tf class for data related ops

# preprocess got data using pickle handler class
gotData = PickleHandler("./got.pkl")

# lenght of sequence to consider for training
len_seq = 100

# total examples per epoch to target
examples_per_seq = len(gotData.text2num) // len_seq

# make a data set using our model
char_dataset = tf.data.Dataset.from_tensor_slices(gotData.text2num)

# make a sequence of text datset
sequence = char_dataset.batch(len_seq + 1, drop_remainder=True)


# split data set into two parts
def split_input_output(chunk):
    input_text = chunk[:-1]  # input of the text
    output_text = chunk[1:]  # output of the text

    return input_text, output_text


# create a dateset mapping to split data accordingly
dataset = sequence.map(split_input_output)  # map sequence to split dataset

for input_text, output_text in dataset.take(1):
    print(gotData.idx2char[input_text])
    print(gotData.idx2char[output_text])
