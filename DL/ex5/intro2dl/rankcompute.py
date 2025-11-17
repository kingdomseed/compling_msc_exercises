import numpy as np

A = np.array([[2, 1, 5],
              [1, 3, 1],
              [3, 4, 6]])
rank = np.linalg.matrix_rank(A)
print("Rank of the matrix A is:", rank)