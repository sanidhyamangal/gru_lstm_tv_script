from pickle_handler import PickleHandler  # to handle pickle files
import tensorflow as tf  # for deep learning libs
import os  # for os related work

# load data set to work
gotData = PickleHandler("./got.pkl")

# to set seq lenght for the learning
len_seq = 1000

# total num of examples
examples_per_seq = len(gotData.text2num) // len_seq

# to make a char dataset
char_dataset = tf.data.Dataset.from_tensor_slices(gotData.text2num)

# design a sequence of tensor from this char data
sequences = char_dataset.batch(len_seq + 1, drop_remainder=True)


def split_input_output(chunk):
    input_text = chunk[:-1]  # input of the text
    output_text = chunk[1:]  # output of the text

    return input_text, output_text


# prepare dataset of these sequences from this split function
dataset = sequences.map(split_input_output)

# define some dataset params
BATCH_SIZE = 64

# buffer size for storage
BUFFER_SIZE = 1000

# shuffle dataset for our model
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# some hyper prams
rnn_units = 1024  # number of rnn units

embedding_units = 256  # size of embedding

vocab_size = gotData.vocab_size()  # vocab size for input

# build model function
def build_model(vocab_size, rnn_units, embedding_units, batch_size):
    # make a sequenctial model
    model = tf.keras.Sequential(
        [
            # first embedding layer
            tf.keras.layers.Embedding(
                vocab_size, embedding_units, batch_input_shape=[batch_size, None]
            ),
            # hidden gru layer
            tf.keras.layers.GRU(
                rnn_units,
                return_sequences=True,
                stateful=True,
                recurrent_initializer="glorot_uniform",
            ),
            # drop out layer
            tf.keras.layers.Dropout(0.4),
            # dense layer
            tf.keras.layers.Dense(vocab_size),
        ]
    )

    return model


# develop a model for our inputs
model = build_model(vocab_size, rnn_units, embedding_units, BATCH_SIZE)

# loss function
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True
    )


# compile model
model.compile(optimizer="rmsprop", loss=loss)

# print summary of this model
print(model.summary())

# define epochs to run on the data
EPOCHS = 10

# model checkpoints
checkpoint_dir = "./tranining_checkpoints_gru"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# create a callback for this dir
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_prefix)

# create a history of this model
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
