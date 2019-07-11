# load tensorflow for deep learning
import tensorflow as tf
import numpy as np  # for matrix multiplication
from pickle_handler import PickleHandler  # handle pickle data
from sys import argv
from tqdm import tqdm

filename, modelfile, outputfile = argv


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
        tf.keras.layers.GRU(
            1024,
            stateful=True,
            return_sequences=True,
            recurrent_initializer="glorot_uniform",
        ),
        # dropout layer
        tf.keras.layers.Dropout(0.4),
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
    temperature = 1.0

    # reset all the states of model
    model.reset_states()

    # iterate into negerate
    for i in tqdm(range(num_generate), ncols=100):
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

        # if i % 1000 == 0:
        #     print("Generated {}% of string".format(i / 100))
    return string_input + "".join(text_generated)


with open(outputfile, 'w', encoding='utf-8') as fp:
    text = generator_function(model, u"JON: ")
    fp.write(text)