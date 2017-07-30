from colorsys import rgb_to_hsv
import os
import numpy as np
from keras.utils import get_file


def load_sherwin_colors():
    """
    Download & format the Sherwin dataset
    """
    def process_line(line):
        l = line.strip().split('\t')

        _, name, r, g, b = l

        h, s, v = rgb_to_hsv(
            int(r) / 255.0,
            int(g) / 255.0,
            int(b) / 255.0
        )

        return name, h, s, v

    text = get_file('sherwin.txt', 'https://raw.githubusercontent.com/devinplatt/paint-color-names/master/sherwin.txt')

    lines = np.array([process_line(line) for line in open(text)])

    return lines


def invert_dict(kv):
    return {v: k for k, v in kv.items()}


def load_glove_embeddings(root_dir='http://nlp.stanford.edu/data/',
                          dataset='glove.6B',
                          dimensions=100):
    """Download pre-trained GloVe word vectors
    https://nlp.stanford.edu/projects/glove/

    this might break with things other than 'glove.6b' right now

    # Arguments
        root_dir: where the data is stored.
        dataset: these seems to be grouped by number of tokens
            options are 'glove.6d', 'glove.twitter.27B', 'glove.840B.300d',
            and 'glove.42B.300d'.
            defaults to'glove.6B'
        dimensions: the number of dimension to load. depends on the dataset

    # Returns
        An embeddings index

    """

    fname = dataset + '.zip'

    txt_fname = dataset + '.' + str(dimensions) + 'd' + '.txt'

    txt_path = os.path.join(os.path.expanduser('~/.keras/'),
                            dataset,
                            txt_fname)

    if not os.path.exists(txt_path):
        f = get_file(fname,
                     origin=root_dir + fname,
                     extract=True,
                     cache_subdir=dataset)

    embeddings_index = {}

    with open(txt_path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    return embeddings_index
