# load tensorflow for deep learning
import tensorflow as tf
import numpy as np  # for matrix multiplication
from pickle_handler import PickleHandler  # handle pickle data
from sys import argv

filename, modelfile, output_file = argv


# load gotdata
gotData = PickleHandler("./got.pkl")


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True
    )


# define a new model
model = tf.keras.Sequential(
    [
        # a embedding layer
        tf.keras.layers.Embedding(
            gotData.vocab_size(), 256, batch_input_shape=[1, None]
        ),
        # lstm layer
        # bidirectional rnn layer
        tf.keras.layers.Bidirectional(
            tf.keras.layers.SimpleRNN(
                1024,
                return_sequences=True,
                stateful=True,
                recurrent_initializer="glorot_uniform",
            )
        ),
        # dropout layer
        tf.keras.layers.Dropout(0.4),
        # dense layer
        tf.keras.layers.Dense(gotData.vocab_size()),
    ]
)

# load model weights
model.load_weights(modelfile)

print("priniting model summary....")
print(model.summary())

# generator function
def generator_function(model, string_input):

    # num of chars to generate
    num_generate = 1000

    input_val = [gotData.char2idx[s] for s in string_input]
    input_val = tf.expand_dims(input_val, 0)

    # set a empty generator list
    text_generated = []

    # temperature for our prediction
    temperature = 1e-3

    # reset all the states of model
    model.reset_states()

    # iterate into negerate
    for i in range(num_generate):
        # get the predictions
        predictions = model(input_val)

        # remove the batch dims
        predictions = tf.squeeze(predictions, 0)

        # using categorial data for the predictions
        predictions = predictions / temperature
        prediction_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # pass the hidden current output to model as an input along with the hidden state
        input_val = tf.expand_dims([prediction_id], 0)

        # append into text generated
        text_generated.append(gotData.idx2char[prediction_id])

        if i % 100 == 0:
            print("Generated {}% of string".format(i / 10))
    return string_input + "".join(text_generated)


print(generator_function(model, "Jon: "))