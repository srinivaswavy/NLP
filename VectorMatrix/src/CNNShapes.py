import numpy as np
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, SimpleRNN, Dense

N = 4
T = 10
D = 3
K = 2
M = 5

X = np.random.randn(N, T, D)
print(X)

i = Input((T, D))
x = SimpleRNN(M)(i)
x = Dense(K)(x)

model = Model(i, x)

print(model.summary())

y = model.predict(X)

print(y)

print(model.weights)

Wx, Wh, bh = model.layers[1].get_weights()

print(Wx.shape, Wh.shape, bh.shape)

Wy, by = model.layers[2].get_weights()

print(Wy.shape, by.shape)

for seq_index in range(X.shape[0]):
    x = X[seq_index]
    h_last = np.zeros(M)
    Y = []
    for t in range(T):
        h = np.tanh(x[t].dot(Wx) + h_last.dot(Wh) + bh)
        y = h.dot(Wy) + by
        h_last = h
        Y.append(y)
    print(Y[-1])


