import numpy as np
from scipy.sparse import csr_matrix

# Create a sparse matrix for demonstration purposes
data = np.array([[0, 1, 0], [2, 0, 3], [4, 0, 5],[0,0,0]])
sparse_matrix = csr_matrix(data)

# mean of non zero elements per row
non_zero_counts_per_row = np.diff(sparse_matrix.indptr)

sums = np.squeeze(np.asarray(sparse_matrix.sum(axis=1)))

average_non_zero_elements_per_row = [ -1 if count==0 else sum/count for sum,count in zip(sums,non_zero_counts_per_row)]

print(average_non_zero_elements_per_row)
