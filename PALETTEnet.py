import numpy as np
# from colorsys import rgb_to_hsv, hsv_to_rgb
from keras.layers import (Input, Dense, Embedding, LSTM, TimeDistributed,
                          concatenate)
from keras.layers.core import RepeatVector
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import get_file
from utils import load_sherwin_colors

class PALETTEnet():

    def __init__(self):
        self.MAX_LEN = None
        self.vocab_size = None
        self.EMBEDDING_DIM = None
        self.embedding_matrix = None
    
    def variable_initializer(self):
        self.EMBEDDING_DIM = 100
    
    def load_weights(weights):
        raise 'NotImplemented'

    def get_word(self, index):
        return self.token_index.get(index)

    def train(self, caption_input, color_input, output, epochs=60):
        """
        Train the model w.r.t. passed inputs & outputs

        # Arguments
            caption_input: FIXME
            color_input: FIXME
            output: FIXME
        
        # Returns
            Keras History object
        """
        model_names = ('color_weights.{epoch:02d}-{val_loss:.2f}.hdf5')
        model_checkpoint = ModelCheckpoint(model_names,
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=False,
                                           save_weights_only=False)

        history = self.model.fit(
            {'caption_input': caption_input, 'color_input': color_input},
            {'output': output},
            callbacks=[model_checkpoint],
            epochs=epochs)

        return history

    def predict(self, hsv, output_length=2):
        text = np.zeros((1, self.MAX_LEN))
        color = np.array([hsv])

        words = []
        for word in range(output_length):
            pred = self.model.predict([text, color])
            w = np.argmax(pred[0, word, :])
            words.append(self.get_word(w))

        return ' '.join(words)


def Pnet(max_token_length, vocabulary_size, embedding_matrix, embedding_dim=100):
    # text model
    caption_input = Input(shape=(max_token_length,),
                          dtype='int32',
                          name='caption_input')
    embedding = Embedding(vocabulary_size,
                          embedding_dim,
                          weights=[embedding_matrix],
                          input_length=max_token_length,
                          trainable=False)(caption_input)
    lstm_1 = LSTM(embedding_dim, return_sequences=True)(embedding)
    tdd = TimeDistributed(Dense(embedding_dim))(lstm_1)

    # color model
    color_input = Input(shape=(3,), name='color_input')
    color_dense = Dense(embedding_dim)(color_input)
    color_repeated = RepeatVector(max_token_length, )(color_dense)

    merged_layers = concatenate([tdd, color_repeated])

    lstm_2 = LSTM(1000, return_sequences=True)(merged_layers)

    output = TimeDistributed(Dense(vocabulary_size, activation='softmax'),
                             name='output')(lstm_2)

    model = Model(inputs=[caption_input, color_input], outputs=output)

    return model



if __name__ == '__main__':
    from keras.utils import plot_model
    model = Pnet(max_token_length=8, vocabulary_size=1024)
    plot_model(model, './images/model.png')
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
