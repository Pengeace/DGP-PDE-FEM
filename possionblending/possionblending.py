import numpy as np
import pyamg
from scipy import sparse

from fem.fem import FiniteElement
from linsolver import sparse_solver


class PossionBlending():
    def __init__(self, source, mask, target, mask_offset, slow_solver=False):
        self.source = source
        self.mask = mask
        self.target = target
        self.mask_offset = mask_offset
        self.slow_solver = slow_solver

    def judge_border(self, p):
        x, y = p
        if not self.mask[x, y]:
            return False
        row, col = self.mask.shape
        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 2):
                if (i >= 0 and i < row) and (j >= 0 and j < col) and ((i, j) != (x, y)):
                    if not self.mask[i, j]:
                        return True
        return False

    def find_points_in_mask(self):
        if len(self.mask.shape) == 3:
            self.mask = self.mask[:, :, 0]
        self.mask = self.mask.astype(np.int) / 255
        nonzero = np.nonzero(self.mask)
        self.mask[self.mask == 0] = False
        self.mask[self.mask != False] = True

        # record all points in Omega and Omega border
        self.point_indexes = list(zip(nonzero[0], nonzero[1]))

        # find border points
        self.border_judge = [False] * len(self.point_indexes)
        for i, p in enumerate(self.point_indexes):
            if self.judge_border(p):
                self.border_judge[i] = True

    def laplace_stencil(self, coord, channel):
        i, j = coord
        source = self.source[:, :, channel]
        val = (4 * source[i, j]) \
              - (1 * source[i + 1, j]) \
              - (1 * source[i - 1, j]) \
              - (1 * source[i, j + 1]) \
              - (1 * source[i, j - 1])
        return val

    def fdm_solver(self):
        print('- FDM solver.')
        self.find_points_in_mask()
        target_rst = np.copy(self.target)

        A_rows = []
        A_cols = []
        A_data = []
        for i, p in enumerate(self.point_indexes):
            x, y = p
            if self.border_judge[i]:
                A_rows.append(i)
                A_cols.append(i)
                A_data.append(1)
            else:
                A_rows.append(i)
                A_cols.append(i)
                A_data.append(4)
                for p_adj in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                    j = self.point_indexes.index(p_adj)
                    A_rows.append(i)
                    A_cols.append(j)
                    A_data.append(-1)

        A = sparse.coo_matrix((A_data, (A_rows, A_cols))).tocsr()

        # 3 channels
        for ch in range(self.target.shape[2]):
            b = np.zeros(len(self.point_indexes))
            for i, p in enumerate(self.point_indexes):
                x, y = p
                if self.border_judge[i]:
                    b[i] = self.target[x + self.mask_offset[0], y + self.mask_offset[1], ch]
                else:
                    b[i] = self.laplace_stencil(p, ch)

            print('Solving...')
            if not self.slow_solver:
                X = pyamg.solve(A, b, verb=False, tol=1e-10)
            else:
                X = sparse_solver.sparse_gauss_seidel([A_rows, A_cols, A_data], b, max_iter_time=20000,
                                                      sparse_input=True)
            print("End one channel.")

            X[X > 255] = 255
            X[X < 0] = 0
            X = np.array(X, self.target.dtype)
            for i, p in enumerate(self.point_indexes):
                x, y = p
                target_rst[x + self.mask_offset[0], y + self.mask_offset[1], ch] = X[i]

        return target_rst

    def fem_solver(self):
        print('- FEM solver.')
        self.find_points_in_mask()
        target_rst = np.copy(self.target)

        # 3 channels
        fem = None
        for ch in range(self.target.shape[2]):

            border_values = []
            rhs_dict = dict()
            for i, p in enumerate(self.point_indexes):
                x, y = p
                rhs_dict[tuple(p)] = self.laplace_stencil(p, ch)
                if self.border_judge[i]:
                    border_values.append((i, self.target[x + self.mask_offset[0], y + self.mask_offset[1], ch]))

            print('Solving...')
            if not fem:
                fem = FiniteElement(self.point_indexes, border_values, A=np.array([[1, 0], [0, 1]]), func=rhs_dict, q=0,
                                    slow_solver=self.slow_solver)
            else:
                fem.update_border_and_func(border_values, rhs_dict)
            fem.solve()
            X = fem.solution
            print("End one channel.")

            X[X > 255] = 255
            X[X < 0] = 0
            X = np.array(X, self.target.dtype)
            for i, p in enumerate(self.point_indexes):
                x, y = p
                target_rst[x + self.mask_offset[0], y + self.mask_offset[1], ch] = X[i]

        return target_rst

    def direct_solver(self):
        print('- Direct paste.')
        self.find_points_in_mask()
        target_rst = np.copy(self.target)

        # 3 channels
        for ch in range(self.target.shape[2]):
            for i, p in enumerate(self.point_indexes):
                x, y = p
                target_rst[x + self.mask_offset[0], y + self.mask_offset[1], ch] = self.source[x, y, ch]

        return target_rst
