# import pickle # to load pickle data
import numpy as np  # for matrix multiplication
from pickle_handler import PickleHandler
import tensorflow as tf  # main tf class for data related ops
import os  # for os related tasks

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
dataset = sequence.map(split_input_output)

# hyperparms for model input data
BATCH_SIZE = 64  # batch size for our model

BUFFER_SIZE = 1000  # buffer size to shuffle dataset

# shuffle dataset for our model
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# model params handeling
# Length of the vocabulary in chars
vocab_size = len(gotData.vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

# function to build a model
def build_model(vocab_size, embedding_dims, rnn_units, batch_size):
    model = tf.keras.Sequential(
        [
            # embedding layer to enhance input dims
            tf.keras.layers.Embedding(
                vocab_size, embedding_dims, batch_input_shape=[BATCH_SIZE, None]
            ),
            # bidirectional rnn layer
            tf.keras.layers.Bidirectional(
                tf.keras.layers.SimpleRNN(
                    rnn_units,
                    return_sequences=True,
                    stateful=True,
                    recurrent_initializer="glorot_uniform",
                )
            ),
            # drop out layer for better training efficiency
            tf.keras.layers.Dropout(rate=0.4),
            # dense layer to connect them
            tf.keras.layers.Dense(vocab_size),
        ]
    )

    # return model
    return model


model = build_model(
    vocab_size=vocab_size,
    embedding_dims=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE,
)

# loss function
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True
    )


# compile model
model.compile(optimizer="rmsprop", loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = "./training_checkpoints"
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# checkpoints callbacks
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix, save_weights_only=True
)

EPOCHS = 10

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
