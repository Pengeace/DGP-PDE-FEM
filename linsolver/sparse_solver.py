import time

import numpy as np


def sparse_jacobi(A, b, sparse_input=False, max_iter_time=30000, min_iter_time=500, tolerance=1e-10):
    """
    Jacobi iteration (with column compressed sparse matrix method) for solving the linear equations with below form:
        A * u = b

    ---------------
    :param A: (N, N) array_like if sparse_input || (RowIndexes, ColumnIndexes, Data) if not sparse_input
        The input data of matrix A.
    :param b: (N, 1) array_like
        Vector b (n*1 size). Input data for the right hand side.
    :param sparse_input: bool, optional
        Whether the input A is sparse format or not.
    :param max_iter_time: int, optional
        Maximal iteration times.
    :param min_iter_time: int, optional
        Minimal iteration times
    :param tolerance: float, optional
        The difference tolerance for adjacent u solutions.
    ---------------
    :return:
        The solution array u. Or suggestive error message.
    """
    start = time.clock()

    b = np.array(b)
    N = b.shape[0]

    # pre-process
    if sparse_input:
        # the input data is sparse represented, that is, just store the nonzero items in usual matrix 'A'
        if max(A[0]) >= N or max(A[1]) >= N:
            return "The size of matrix 'A' is not corresponding to vector 'b'."
        matrix_col = [0] * N
        matrix_data = [0] * N
        aii = np.zeros(N)
        for i, r in enumerate(A[0]):
            if matrix_col[r] == 0:
                matrix_col[r] = []
                matrix_data[r] = []
            if A[0][i] == A[1][i]:
                aii[r] = A[2][i]
            else:
                matrix_col[r].append(A[1][i])
                matrix_data[r].append(A[2][i])
        for r in range(N):
            matrix_col[r] = np.array(matrix_col[r])
            matrix_data[r] = np.array(matrix_data[r])
    else:
        # the input data is usual matrix format
        A = np.array(A)
        n, m = A.shape
        if n != m:
            return "The row size of matrix 'A' is not equal to its column size."
        if n != N:
            return "The row size of matrix 'A' is not corresponding to the size of vector 'b'."
        matrix_col = [0] * N
        matrix_data = [0] * N
        aii = np.zeros(N)
        for r in range(N):
            aii[r] = A[r, r]
            A[r, r] = 0
            matrix_col[r] = np.nonzero(A[r, :])[0].astype(np.int)
            matrix_data[r] = A[r, matrix_col[r]]
    matrix_col = np.array(matrix_col)
    matrix_data = np.array(matrix_data)
    x = np.zeros(N)
    x_next = np.zeros(N)

    # perform iterative calculation
    for i in range(max_iter_time):
        # update x
        for r in range(N):
            if len(matrix_col[r]):
                x_next[r] = (b[r] - np.dot(x[matrix_col[r]], matrix_data[r])) / aii[r]

        # check terminal condition
        if max(abs(x_next - x)) < tolerance and i > min_iter_time:
            print('Jacobi Iterations (Sparse):', i + 1)
            print('Time past:', time.clock() - start)
            return x_next
        else:
            x_next, x = x, x_next

    print("Maximal iteration times reached.")
    print('Jacobi Iterations (Sparse):', i + 1)
    print('Time past:', time.clock() - start)
    return x_next


def sparse_gauss_seidel(A, b, sparse_input=False, max_iter_time=30000, min_iter_time=500, tolerance=1e-10):
    """
    Gauss-Seidel iteration (with column compressed sparse matrix method) for solving the linear equations with below form:
        A * u = b

    ---------------
    :param A: (N, N) array_like if sparse_input || (RowIndexes, ColumnIndexes, Data) if not sparse_input
        The input data of matrix A.
    :param b: (N, 1) array_like
        Vector b (n*1 size). Input data for the right hand side.
    :param sparse_input: bool, optional
        Whether the input A is sparse format or not.
    :param max_iter_time: int, optional
        Maximal iteration times.
    :param min_iter_time: int, optional
        Minimal iteration times
    :param tolerance: float, optional
        The difference tolerance for adjacent u solutions.
    ----------------
    :return:
        The solution array u. Or suggestive error message.
    """
    start = time.clock()

    b = np.array(b)
    N = b.shape[0]

    # pre-process
    if sparse_input:
        # the input data is sparse represented, that is, just store the nonzero items in usual matrix 'A'
        if max(A[0]) >= N or max(A[1]) >= N:
            return "The size of matrix 'A' is not corresponding to vector 'b'."
        matrix_col = [0] * N
        matrix_data = [0] * N
        aii = np.zeros(N)
        for i, r in enumerate(A[0]):
            if matrix_col[r] == 0:
                matrix_col[r] = []
                matrix_data[r] = []
            if A[0][i] == A[1][i]:
                aii[r] += A[2][i]
            else:
                matrix_col[r].append(A[1][i])
                matrix_data[r].append(A[2][i])
        for r in range(N):
            cols = sorted(list(set(matrix_col[r])))
            col_num = len(cols)
            new_matrix_data_r = np.zeros(col_num)
            pos_map = dict(zip(cols, list(range(col_num))))
            for i, c in enumerate(matrix_col[r]):
                new_matrix_data_r[pos_map[c]] += matrix_data[r][i]

            matrix_col[r] = np.array(cols)
            matrix_data[r] = new_matrix_data_r
    else:
        # the input data is usual matrix format
        # A = np.array(A)
        n, m = A.shape
        if n != m:
            return "The row size of matrix 'A' is not equal to its column size."
        if n != N:
            return "The row size of matrix 'A' is not corresponding to the size of vector 'b'."
        matrix_col = [0] * N
        matrix_data = [0] * N
        aii = np.zeros(N)
        for r in range(N):
            aii[r] = A[r, r]
            A[r, r] = 0
            matrix_col[r] = np.nonzero(A[r, :])[0].astype(np.int)
            matrix_data[r] = A[r, matrix_col[r]]

    matrix_col = np.array(matrix_col)
    matrix_data = np.array(matrix_data)
    x = np.zeros(N)
    x_next = np.zeros(N)

    # perform iterative calculation
    for i in range(max_iter_time):
        # update x
        for r in range(N):
            if len(matrix_col[r]):
                x_next[r] = (b[r] - np.dot(x_next[matrix_col[r]], matrix_data[r])) / aii[r]

        # check terminal condition
        if max(abs(x_next - x)) < tolerance and i > min_iter_time:
            print('Gauss-Seidel Iterations (Sparse):', i + 1)
            print('Time past:', time.clock() - start)
            return x_next
        else:
            x = np.copy(x_next)

    print("Maximal iteration times reached.")
    print('Gauss-Seidel Iterations (Sparse):', i + 1)
    print('Time past:', time.clock() - start)
    return x_next


# just for simple test
if __name__ == '__main__':
    from scipy.linalg import solve

    A = np.array([[8, -3, 2], [4, 11, -1], [6, 3, 12]])
    b = np.array([[20], [33], [36]])

    s1 = solve(A, b).reshape(len(b))

    AA = [[], [], []]
    for row in range(3):
        for column in range(3):
            AA[0].append(row)
            AA[1].append(column)
            AA[2].append(A[row, column])

    s2 = sparse_jacobi(AA, b, sparse_input=True)
    s3 = sparse_gauss_seidel(AA, b, sparse_input=True)

    print(s1, s2, s3)
    print(np.array(s1) - np.array(s2))
