import time

import numpy as np


def jacobi(A, b, max_iter_time=5000, min_iter_time=100, tolerance=1e-10):
    """
    Jacobi iteration for solving the linear equations with below form:
        A * u = b

    ---------------
    :param A: (N, N) array_like
        Matrix A (n*n size). Square input data.
    :param b: (N, 1) array_like
        Vector b (n*1 size). Input data for the right hand side.
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
    end = time.clock()

    A = np.array(A)
    b = np.array(b)
    n, m = A.shape
    if n != m:
        return "The row size of matrix 'A' is not equal to its column size."
    if n != b.shape[0]:
        return "The row size of matrix 'A' is not equal to the size of vector 'b'."
    b.reshape(n)
    aii = np.array([A[i, i] for i in range(n)])
    x_before = np.zeros(n)
    x_cur = np.zeros(n)

    # iteration
    for i in range(max_iter_time):
        for r in range(n):
            remain = np.dot(A[r, :], x_before[:]) - A[r, r] * x_before[r]
            x_cur[r] = (b[r] - remain) / aii[r]
        if max(abs(x_cur - x_before)) < tolerance and i > min_iter_time:
            print('Jacobi Iteration:', i + 1)
            print('Time past:', time.clock() - start)
            return x_cur
        else:
            x_cur, x_before = x_before, x_cur
    print("Maximal iteration times reached.")

    print('Jacobi Iteration:', i + 1)
    print('Time past:', time.clock() - start)
    return x_cur


def gauss_seidel(A, b, max_iter_time=5000, min_iter_time=100, tolerance=1e-10):
    """
    Gauss-Seidel iteration for solving the linear equations with below form:
        A * u = b

    ---------------
    :param A: (N, N) array_like
        Matrix A (n*n size). Square input data.
    :param b: (N, 1) array_like
        Vector b (n*1 size). Input data for the right hand side.
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

    A = np.array(A)
    b = np.array(b)
    n, m = A.shape
    if n != m:
        return "The row size of matrix 'A' is not equal to its column size."
    if n != b.shape[0]:
        return "The row size of matrix 'A' is not equal to the size of vector 'b'."
    b.reshape(n)
    aii = np.array([A[i, i] for i in range(n)])
    x_before = np.zeros(n)
    x_cur = np.zeros(n)

    # iteration
    for i in range(max_iter_time):
        for r in range(n):
            remain = np.dot(A[r, :], x_cur[:]) - A[r, r] * x_cur[r]
            x_cur[r] = (b[r] - remain) / aii[r]
        if max(abs(x_cur - x_before)) < tolerance and i > min_iter_time:
            print('Gauss-Seidel Iterations:', i + 1)
            print('Time past:', time.clock() - start)
            return x_cur

        x_before = np.copy(x_cur)
    print("Maximal iteration times reached.")
    print('Gauss-Seidel Iterations:', i + 1)
    print('Time past:', time.clock() - start)
    return x_cur


# just for simple test
if __name__ == '__main__':
    from scipy.linalg import solve

    A = np.array([[8, -3, 2], [4, 11, -1], [6, 3, 12]])
    b = np.array([[20], [33], [36]])

    s1 = solve(A, b).reshape(len(b))
    s2 = jacobi(A, b)
    s3 = gauss_seidel(A, b)

    print(s1, s2, s3)
    print(np.array(s1) - np.array(s2))
