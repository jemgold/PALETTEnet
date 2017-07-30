from models import Pnet
from generator import Generator
from keras.callbacks import ModelCheckpoint
from utils import load_sherwin_colors

EPOCHS = 60
BATCH_SIZE = 16

colors = load_sherwin_colors()

generator = Generator(colors)

model = Pnet(max_token_length=generator.MAX_TOKEN_LENGTH,
             vocabulary_size=generator.VOCABULARY_SIZE,
             embedding_matrix=generator.embedding_matrix)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

model_names = ('color_weights.{epoch:02d}-{val_loss:.2f}.hdf5')
model_checkpoint = ModelCheckpoint(model_names,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=False,
                                   save_weights_only=False)

history = model.fit_generator(generator=generator.flow(),
                              callbacks=[model_checkpoint],
                              epochs=EPOCHS)
# {'caption_input': caption_input, 'color_input': color_input},
# {'output': output},
