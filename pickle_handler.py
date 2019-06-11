import pickle  # to load pickle file
import numpy as np  # for matix multiplication


class PickleHandler:
    # init function of our class
    def __init__(self, pickle_file):
        # open pickle file for getting input
        with open(pickle_file, "rb") as fp:
            # store pickle data in form of complete text
            self.__complete_text = pickle.load(fp)

        # generate vocab for our text
        # self.__vocab = sorted(set(self.__complete_text))

        # generate char2idx dict
        self.char2idx = dict((c, i) for i, c in enumerate(self.vocab))

        # generate idx2char list
        self.idx2char = np.array(self.vocab)

    # function to return vocab of the text as a text
    @property
    def vocab(self):
        return sorted(set(self.__complete_text))

    # property attribut for the text 2 int function
    @property
    def text2num(self):
        # convert value of text into int mapping
        return np.array([self.char2idx[t] for t in self.__complete_text])

    # function to return vocab size
    def vocab_size(self):
        return len(self.vocab)
