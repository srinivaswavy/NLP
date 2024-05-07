import math
import random

import gensim.downloader as api
import tensorflow
from tensorflow import keras
from keras.layers import Input, Dense, Embedding, Lambda
from keras.models import Model
from keras.preprocessing.text import Tokenizer
import random as rd
import numpy as np
import matplotlib.pyplot as plt

dataset = api.load("text8")

i = 1
for d in dataset:
    print(d)
    i += 1
    if i == 5:
        break

seq_lengths = []
for d in dataset:
    seq_lengths.append(len(d))

print(len(seq_lengths))
plt.hist(seq_lengths)
plt.show()

print("mean:", np.mean(seq_lengths))
print("std dev:", np.std(seq_lengths))
vocabulary_size = 20000
tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(dataset)
sequences = tokenizer.texts_to_sequences(dataset)

print(sequences[0:2])
print(len(tokenizer.word_index))
print(tokenizer.word_index)
print(tokenizer.index_word)

context_size = 10
half_context_size = context_size // 2

embedding_dimension_size = 50

i = Input(shape=(context_size,))
e = Embedding(vocabulary_size, embedding_dimension_size)(i)
mean = Lambda(lambda t: tensorflow.reduce_mean(t, axis=1))(e)
x = Dense(vocabulary_size, use_bias=False)(mean)

model = Model(i, x)
print(model.summary())
batch_size = 128


def data_generator(seqs, batch_size=128):
    X_batch = np.zeros((batch_size, context_size))
    Y_batch = np.zeros(batch_size)
    number_of_batches = math.ceil(len(seqs) / batch_size)
    j = 0
    while True:
        random.shuffle(seqs)
        for i in range(number_of_batches):
            batch = seqs[i * batch_size:(i + 1) * batch_size]
            size_of_batch = len(batch)
            for ii in range(len(batch)):
                seq = batch[ii]
                y_position = np.random.randint(half_context_size, len(seq) - half_context_size - 1)
                x1 = seq[y_position - half_context_size:y_position]
                x2 = seq[y_position + 1:y_position + half_context_size + 1]
                X_batch[ii, :half_context_size] = x1
                X_batch[ii, half_context_size:context_size] = x2
                Y_batch[ii] = seq[y_position]
        yield X_batch[:size_of_batch], Y_batch[:size_of_batch]


model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)

r = model.fit(
    data_generator(sequences),
    epochs=10000,
    steps_per_epoch=int(np.ceil(len(sequences) / batch_size))
)

plt.plot(r.history['loss'], label='loss')
plt.legend()

plt.show()

plt.plot(r.history['accuracy'], label='acc')
plt.legend()
plt.show()

embeddings = model.layers[1].get_weights()[0]
print(embeddings)

from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
neighbors.fit(embeddings)

queen_idx = tokenizer.word_index['queen']
queen = embeddings[queen_idx:queen_idx + 1]
distances, indices = neighbors.kneighbors(queen)

for idx in indices[0]:
    word = tokenizer.index_word[idx]
    print(word)

# man-king = woman-queen

man = embeddings[tokenizer.word_index["man"]]
king = embeddings[tokenizer.word_index["king"]]
queen = embeddings[tokenizer.word_index["queen"]]

distances, indices = neighbors.kneighbors(man - king + queen)

print("Analogy test: ")
for idx in indices[0]:
    word = tokenizer.index_word[idx]
    print(word)

model.save("WordEmbeddings")
