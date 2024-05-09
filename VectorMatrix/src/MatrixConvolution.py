import numpy as np

matrix = np.zeros((10, 10))
convolve_filter = np.zeros((2, 2))

np.random.seed(42)

valid_output = np.zeros(
    (matrix.shape[0] - convolve_filter.shape[0] + 1, matrix.shape[1] - convolve_filter.shape[1] + 1))


def fill_random_values(m):
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            m[i][j] = np.random.randint(0, 10)


fill_random_values(matrix)
print(matrix)

fill_random_values(convolve_filter)
print(convolve_filter)


def convolve(o, m, f):
    filter_row_count = f.shape[0]
    filter_column_count = f.shape[1]

    for i in range(o.shape[0]):
        for j in range(o.shape[1]):
            o[i, j] = m[i:i + filter_row_count, j:j + filter_column_count].flatten().transpose().dot(
                f.flatten())


convolve(valid_output, matrix, convolve_filter)

print(valid_output)
matrix[0:2, 0:2].flatten().transpose().dot(convolve_filter.flatten())

same_output = np.zeros(matrix.shape)
padded_matrix = np.zeros(
    (matrix.shape[0] + convolve_filter.shape[0] - 1, matrix.shape[1] + convolve_filter.shape[1] - 1))
padded_matrix[convolve_filter.shape[0] - 1:, convolve_filter.shape[1] - 1:] = matrix

print(padded_matrix)

convolve(same_output, padded_matrix, convolve_filter)

print(same_output)

full_output = np.zeros(
    (matrix.shape[0] + convolve_filter.shape[0] - 1, matrix.shape[1] + convolve_filter.shape[1] - 1))
padded_matrix = np.zeros((matrix.shape[0] + 2 * (convolve_filter.shape[0] - 1),
                         matrix.shape[1] + 2 * (convolve_filter.shape[1] - 1)))
padded_matrix[convolve_filter.shape[0] - 1: matrix.shape[0] + convolve_filter.shape[0] - 1,
convolve_filter.shape[1] - 1:matrix.shape[1] + convolve_filter.shape[1] - 1] = matrix

print(padded_matrix)

convolve(full_output, padded_matrix, convolve_filter)

print(full_output)
