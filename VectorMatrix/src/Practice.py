from tensorflow import keras
from keras.layers import Embedding
import numpy as np

# model = keras.Sequential()
# model.add(Embedding(1000, 64, input_length=10))
# # The model will take as input an integer matrix of size (batch,
# # input_length), and the largest integer (i.e. word index) in the input
# # should be no larger than 999 (vocabulary size).
# # Now model.output_shape is (None, 10, 64), where `None` is the batch
# # dimension.
# input_array = np.random.randint(1000, size=(32, 10))
# print("input_array:\n", input_array)
# model.compile('rmsprop', 'mse')
# output_array = model.predict(input_array)
# print(output_array.shape)





x = np.random.rand(4, 10, 128)

print(x.shape)

print(x[0].shape)

print(x[0][0].shape)

print(x[0][0][0])