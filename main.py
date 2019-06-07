import pickle # to load pickle data
import numpy as np # for matrix multiplication
# from keras.layers import LSTM, Activation, Dropout, Dense, Input # layers to make models 
# from keras.models import Model, Sequential # model class to make a model 
# from keras.callbacks import TensorBoard # for tensorboard ops 

import tensorflow as tf

# enable eager execution 
tf.enable_eager_execution()

# load season1 pkl data 
with open("season1.pkl", "rb") as pf:
    text = pickle.load(pf)

# make a set of chars 
vocab = sorted(set(text))

# char2idx dict 
char2idx = dict((c,i) for i,c in enumerate(vocab))

# idx2char dic mapping 
idx2char = dict((i,c) for i, c in enumerate(vocab))
# idx2char = np.array(vocab)
# print(idx2char)

# np array for text2num
text2int  = np.array([char2idx[t] for t in text])

# print("{} chars mapped to int {}".format(text[:13], text2int[:13]))

# lenght of sequence to consider for training  
len_seq = 100

# total examples per epoch to target 
examples_per_seq = len(text2int) // len_seq

# make a data set using our model 
char_dataset = tf.data.Dataset.from_tensor_slices(text2int)

# make a sequence of text datset 
sequence = char_dataset.batch(len_seq +1, drop_remainder=True)


# split data set into two parts 
def split_input_output(chunk):
    input_text = chunk[:-1] # input of the text 
    output_text = chunk[1:] # output of the text

    return input_text, output_text


# create a dateset mapping to split data accordingly
dataset = sequence.map(split_input_output) # map sequence to split dataset

# for input_txt, output_txt in dataset.take(1):
#     # for i in range(input_txt.numpy()):
#     print(idx2char[input_txt.numpy()])
    # print("Input data {}".format(repr(' '.join(idx2char[i] for i in range(input_txt.numpy())))))
    # print("Input data {}".format(repr(' '.join(idx2char[i] for i in range(output_txt.numpy())))))
    # print("Output data {}".format(repr(''.join(idx2char[output_txt.numpy()]))))
    
# # splitting data into two parts
# input_txt, output_txt = split_input_output(text2int[:225])

# for i in range(len(input_txt)):
#     print("Input({}):".format(idx2char[input_txt[i]]))
#     print("\tOutput({}):".format(idx2char[output_txt[i]]))
