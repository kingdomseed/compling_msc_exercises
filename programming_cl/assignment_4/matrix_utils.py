
# Takes a list of lists of equal length 
# and returns the diagonal
# matrix is a square matrix - a list of n sublists
# where each sublist has n elements
# return the diagonal of the matrix as a list
def diagonal(matrix: list[list]):
    diagonal = []
    for i in range(len(matrix)):
        diagonal.append(matrix[i][i])
    return diagonal


# Transposes a matrix
# matrix is a list of lists of equal length
# return the transpose of the matrix as a list of lists

def transpose(matrix: list[list]):

    row_len = len(matrix[0])
    col_len = len(matrix)
    transposed = [[] for _ in range(row_len)]
    for column in range(row_len):
        for cell in range(col_len):
            transposed[column].append(matrix[cell][column])
    return transposed
