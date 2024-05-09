import numpy as np

matrix = np.zeros((10, 10))
convolve_filter = np.zeros((2, 2))

np.random.seed(42)

output = np.zeros((matrix.shape[0] - convolve_filter.shape[0] + 1, matrix.shape[1] - convolve_filter.shape[1] + 1))


def fill_random_values(m):
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            m[i][j] = np.random.randint(0, 10)


fill_random_values(matrix)
print(matrix)

fill_random_values(convolve_filter)
print(convolve_filter)

filter_row_count = convolve_filter.shape[0]
filter_column_count = convolve_filter.shape[1]

for i in range(output.shape[0]):
    for j in range(output.shape[1]):
        output[i, j] = matrix[i:i + filter_row_count, j:j + filter_column_count].flatten().transpose().dot(
            convolve_filter.flatten())

print(output)
matrix[0:2, 0:2].flatten().transpose().dot(convolve_filter.flatten())
