import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import random
from utils import invert_dict, load_glove_embeddings


class Generator():
    """
    Data generator for the palette model
    """
    def __init__(self, dataset, embedding_dim=100):
        self.num_words = 10000  # number of words to tokenize
        self.MAX_TOKEN_LENGTH = 3
        self.VOCABULARY_SIZE = None

        self.process_dataset(dataset)
        self.make_embedding_matrix(embedding_dim)

    def process_dataset(self, data):
        texts = data[:, 0]
        self.input_color_data = data[:, 1:]

        tokenizer = Tokenizer(num_words=self.num_words, )
        tokenizer.fit_on_texts(texts)

        sequences = tokenizer.texts_to_sequences(texts)

        self.word_index = tokenizer.word_index
        self.token_index = invert_dict(self.word_index)

        print('Found {} unique tokens.'.format(len(self.word_index)))

        self.input_captions = pad_sequences(sequences, maxlen=3)

        y = np.zeros((self.input_captions.shape[0],
                      self.MAX_TOKEN_LENGTH,
                      len(self.word_index) + 1))
        
        for i, sequence in enumerate(sequences):
            for j, w in enumerate(sequence):
                y[i, j, w] = 1
        
        self.output_captions = y

    def get_data(self):
        return self.wrap_in_dictionary(one_hot_caption=self.input_captions,
                                       color_features=self.input_color_data,
                                       one_hot_target=self.output_captions)

    def flow(self, batch_size=32, shuffle=True):
        batch_captions = np.zeros # self.input_captions.shape(turn first dim into batch_size)
        batch_colors = np.zeros # self.input_color_data.shape(turn first dim into batch_size)
        batch_output = np.zeros # self.output_captions (first dim to batch_szie)

        while True:
            for i in range(batch_size):
                if shuffle:
                    index = random.choice(range(len(batch_captions)))
                else:
                    index = i
                batch_captions[i] = self.input_captions[index]
                batch_colors[i] = self.input_color_data[index]
                batch_output[i] = self.output_captions[index]
            
        yield self.wrap_in_dictionary()
        raise NotImplementedError()

    def make_embedding_matrix(self, embedding_dim):
        embeddings_index = load_glove_embeddings()

        print('Found {} word vectors.'.format(len(embeddings_index)))

        embedding_matrix = np.zeros((len(self.word_index) + 1, embedding_dim))

        not_found = []
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is None:
                not_found.append(word)
            else:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def wrap_in_dictionary(self, one_hot_caption, color_features,
                           one_hot_target):
        return [{'caption_input': one_hot_caption,
                 'color_input': color_features},
                {'output': one_hot_target}]
