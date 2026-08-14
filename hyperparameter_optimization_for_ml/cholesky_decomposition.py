# Python program to find decompose of a
# matrix using Cholesky Decomposition

def choleskyDecomposition(matrix):
    n = len(matrix)

    # to store the lower triangular matrix
    lower = [[0] * n for _ in range(n)]

    # Decomposing a matrix into Lower Triangular
    for i in range(n):
        for j in range(i + 1):
            sum = 0

            # summation for diagonals
            if j == i:
                for k in range(j):
                    sum += lower[j][k]**2
                lower[j][j] = int((matrix[j][j] - sum)**0.5)
            else:

                # Evaluating L(i, j) using L(j, j)
                for k in range(j):
                    sum += lower[i][k] * lower[j][k]
                lower[i][j] = (matrix[i][j] - sum) // lower[j][j]

    # Displaying Lower Triangular Matrix
    for i in range(n):
        print(" ".join(map(str, lower[i])))

    print()

    # Displaying Transpose of Lower Triangular Matrix
    for i in range(n):
        print(" ".join(map(str, [lower[j][i] for j in range(n)])))

if __name__ == "__main__":
    matrix = [
        [4, 12, -16],
        [12, 37, -43],
        [-16, -43, 98]
    ]
    choleskyDecomposition(matrix)