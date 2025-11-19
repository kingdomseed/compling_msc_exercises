import matrix_utils
import list_utils


# test the last function
print(list_utils.last([1, 2, 3]))

# test the middle function
print(list_utils.middle([1, 2, 3]))

# test the product function
print(list_utils.product([1, 2, 3]))

# test the mean function
print(list_utils.mean([1, 2, 3]))

# test the even_sum function
print(list_utils.even_sum([1, 2, 3]))

# test the diagonal function
print(matrix_utils.diagonal([[1, 2], [3, 4]]))

# test the transpose function
mat1 = [
        [0, 1, 2, 3, 4], 
        [0, 1, 2, 3, 4]
    ]
print("Trasposed: ", matrix_utils.transpose(mat1))
