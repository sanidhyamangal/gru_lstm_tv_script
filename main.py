# import pickle # to load pickle data
import numpy as np  # for matrix multiplication
from pickle_handler import PickleHandler
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

# hyperparms for model input data
BATCH_SIZE = 64  # batch size for our model

BUFFER_SIZE = 1000  # buffer size to shuffle dataset

# shuffle dataset for our model
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# model params handeling

RNN_UNITS = 1024  # rnn units to use for training
EMBEDDING_DIMS = 256  # to embedd our variables into lower dims
vocab_size = gotData.vocab_size()  # set vocab size for our model

# function to build a model
def build_model(vocab_size, embedding_dims, rnn_units, batch_size):
    model = tf.keras.Sequential(
        [
            # embedding layer to enhance input dims
            tf.keras.layers.Embedding(
                vocab_size, EMBEDDING_DIMS, batch_input_shape=[BATCH_SIZE, None]
            ),
            # lstm layer
            tf.keras.layers.LSTM(
                rnn_units,
                return_sequences=True,
                stateful=True,
                recurrent_initializer="glorot_uniform",
            ),
            # drop out layer for better training efficiency
            tf.keras.layers.Dropout(rate=0.4),
            # dense layer to connect them
            tf.keras.layers.Dense(vocab_size),
        ]
    )

    # return model
    return model


# develop model with parameters
model = build_model(
    vocab_size=vocab_size,
    embedding_dims=EMBEDDING_DIMS,
    rnn_units=RNN_UNITS,
    batch_size=BATCH_SIZE,
)

# print model summary
print(model.summary())
