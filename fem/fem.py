import numpy as np
import pyamg
from scipy import sparse
from scipy.spatial import Delaunay

from linsolver import sparse_solver
from triangulation.delaunay import delaunay


class Element:
    def __init__(self, points, global_indexes, fem):
        self.points = np.array(points)
        self.global_indexes = global_indexes
        self.fem = fem

        self.reference_triangle = np.array([[0, 0], [1., 0], [0, 1.]])
        self.reference_grad = np.array([[-1., -1], [1., 0], [0, 1.]])

    def perform_calculation(self):
        self._calculate_transform()
        self._calculate_stiffness_matrix()
        self._calulate_load_vector()

    def _calculate_transform(self):
        reference_coord = np.array([self.reference_triangle[:, 0], self.reference_triangle[:, 1], [1] * 3])
        transformed_coord = np.array([self.points[:, 0], self.points[:, 1], [1] * 3])

        trans = np.dot(transformed_coord, np.linalg.inv(reference_coord))

        self.transform_matrix = trans[0:-1, 0:-1]

        self.area = abs(np.linalg.det(self.transform_matrix)) / 2

    def _calculate_stiffness_matrix(self):

        transform_matrix_inv = np.linalg.inv(self.transform_matrix)
        self.element_stiffness_matrix = np.zeros((3, 3))

        for row in range(3):
            for col in range(3):
                part_u_left_grad = np.dot(np.dot(self.fem.A, transform_matrix_inv.T), self.reference_grad[row])
                part_u_right_grad = np.dot(transform_matrix_inv.T, self.reference_grad[col])

                part_u_grad = self.area * np.dot(part_u_left_grad, part_u_right_grad)

                part_u = (self.area / 6.0) if row == col else (self.area / 12.0)

                self.element_stiffness_matrix[row, col] = part_u_grad + self.fem.q * part_u

    def _calulate_load_vector(self):

        mean_f = np.mean([self.fem.get_func_value(x) for x in self.points])
        self.element_load_vector = np.array([mean_f * self.area / 3] * 3)


class FiniteElement:
    """
    Finite Element Method to solve the 2D Elliptic Partial Differentiation differential Equation with below form:

        div(A grad(u)) + q u = func

    """

    def __init__(self, points, boundaries, A, q, func, slow_solver=True):
        self.points = np.array(points)
        self.dirichlet_boundaries = np.array(boundaries)
        self.A = A
        self.q = q
        self.f = func
        self.slow_solver = slow_solver

        self.triangles = []
        self.point_num = len(points)

    def solve(self):
        if len(self.triangles) == 0:
            self._get_mesh()
        self._process_each_element()
        self._calculate_global_stiffness_matrix()
        self._calulate_global_load_vector()
        self._deal_with_dirichlet_bound()
        self._solve_linear_equations()

    def update_border_and_func(self, boundaries, func):
        self.dirichlet_boundaries = np.array(boundaries)
        self.f = func

    def get_func_value(self, x):
        if isinstance(self.f, dict):
            return self.f[tuple(x)]
        else:
            return self.f(x)

    def _get_mesh(self):

        if self.slow_solver:
            self.triangles = delaunay(self.points)
        else:
            triangulation = Delaunay(self.points)
            self.triangles = triangulation.simplices

    def _process_each_element(self):
        self.elements = []
        for tri in self.triangles:
            ele = Element(points=[self.points[v] for v in tri], global_indexes=tri, fem=self)
            ele.perform_calculation()
            self.elements.append(ele)

    def _calculate_global_stiffness_matrix(self):

        self.global_stiffness_matrix_row = []
        self.global_stiffness_matrix_col = []
        self.global_stiffness_matrix_data = []

        boundary_indexes = set(self.dirichlet_boundaries[:, 0].astype('int'))

        for ele in self.elements:
            for row in range(3):
                if ele.global_indexes[row] not in boundary_indexes:
                    for col in range(3):
                        self.global_stiffness_matrix_row.append(ele.global_indexes[row])
                        self.global_stiffness_matrix_col.append(ele.global_indexes[col])
                        self.global_stiffness_matrix_data.append(ele.element_stiffness_matrix[row, col])

    def _calulate_global_load_vector(self):

        self.global_load_vector = np.zeros(self.point_num)
        for ele in self.elements:
            for v in range(3):
                self.global_load_vector[ele.global_indexes[v]] += ele.element_load_vector[v]

    def _deal_with_dirichlet_bound(self):
        for index, val in self.dirichlet_boundaries:
            index = int(index)

            self.global_stiffness_matrix_row.append(index)
            self.global_stiffness_matrix_col.append(index)
            self.global_stiffness_matrix_data.append(1)

            self.global_load_vector[index] = val

    def _solve_linear_equations(self):

        if not self.slow_solver:
            self.global_stiffness_matrix_csr = sparse.coo_matrix((self.global_stiffness_matrix_data, (
                self.global_stiffness_matrix_row, self.global_stiffness_matrix_col))).tocsr()
            self.solution = pyamg.solve(self.global_stiffness_matrix_csr, self.global_load_vector, verb=False,
                                        tol=1e-10)
        else:
            global_stiffness_sparse = [np.array(self.global_stiffness_matrix_row),
                                       np.array(self.global_stiffness_matrix_col),
                                       np.array(self.global_stiffness_matrix_data)]
            self.solution = sparse_solver.sparse_gauss_seidel(global_stiffness_sparse, self.global_load_vector,
                                                              sparse_input=True)

            ## these solver methods are for test
            # self.global_stiffness = sparse.coo_matrix((self.global_stiffness_matrix_data, (
            #     self.global_stiffness_matrix_row, self.global_stiffness_matrix_col))).tocsr()
            # self.solution = linsolver.jacobi(self.global_stiffness.toarray(), self.global_load_vector)
            # self.solution = linsolver.gauss_seidel(self.global_stiffness.toarray(), self.global_load_vector)
            # self.solution = sparse_solver.sparse_jacobi(self.global_stiffness.toarray(), self.global_load_vector, sparse_input=False)
            # self.solution = sparse_solver.sparse_gauss_seidel(self.global_stiffness.toarray(), self.global_load_vector, sparse_input=False)

        if isinstance(self.solution, str):
            print("The inputs for linear solver have problems.")
