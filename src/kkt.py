import numpy as np


class KKT():
    def __init__(self, c, A, b, G, h, p, m, n):
        self.c = c.copy()
        self.A = A.copy()
        self.b = b.copy()
        self.G = G.copy()
        self.h = h.copy()

        self.p = p
        self.m = m
        self.n = n

        self.c_shape = c.shape
        self.A_shape = A.shape
        self.b_shape = b.shape
        self.G_shape = G.shape
        self.h_shape = h.shape

        self.zero_nn = np.zeros([self.n, self.n])
        self.zero_pp = np.zeros([self.p, self.p])
        self.zero_mm = np.zeros([self.m, self.m])
        self.zero_11 = np.zeros([1, 1])
        self.zero_mp = np.zeros([self.m, self.p])
        self.zero_n1 = np.zeros([self.n, 1])
        self.zero_p1 = np.zeros([self.p, 1])
        self.zero_m1 = np.zeros([self.m, 1])
        self.zero_1m = self.zero_m1.transpose()
        self.zero_npluspmplus1 = np.zeros([self.n + self.p, self.m + 1])

        self.one_mm = np.eye(self.m)
        self.one_11 = np.eye(1)
        self.one_mplus1mplus1 = np.eye(self.m + 1)
        self.matrix_youshang = np.vstack([self.zero_npluspmplus1, self.one_mplus1mplus1])

        self.KKT_Matrix = self.KKT(c, A, b, G, h)
        self.matrix_shang = np.hstack([-1 * self.KKT_Matrix, self.matrix_youshang])

        self.equ_fenge = [n, p, m, 1, m, 1]
        self.fenge = np.add.accumulate(self.equ_fenge)

    def KKT(self, c, A, b, G, h):
        hang_0 = np.concatenate((self.zero_nn, self.A.transpose(), self.G.transpose(), self.c), axis=1)
        hang_1 = np.concatenate((-1 * self.A, self.zero_pp, self.zero_mp.transpose(), b), axis=1)
        hang_2 = np.concatenate((-1 * self.G, self.zero_mp, self.zero_mm, self.h), axis=1)
        hang_3 = np.concatenate(
            (-1 * self.c.transpose(), -1 * self.b.transpose(), -1 * self.h.transpose(), self.zero_11), axis=1)
        return np.concatenate((hang_0, hang_1, hang_2, hang_3), axis=0)

    def solve(self, equation_right, W1, W2, k, t):
        xiamatrix = np.zeros([self.m + 1, self.n + self.p + self.m + 1 + self.m + 1])
        xiamatrix[0:self.m, self.n + self.p:self.n + self.p + self.m] = W1
        xiamatrix[0:self.m, self.n + self.p + self.m + 1:self.n + self.p + self.m + 1 + self.m] = W2
        xiamatrix[self.m, self.n + self.p + self.m] = k
        xiamatrix[self.m, self.n + self.p + self.m + 1 + self.m] = t
        matrix = np.vstack([self.matrix_shang, xiamatrix])
        direction = np.linalg.solve(matrix, equation_right)
        direction_array = np.vsplit(direction, self.fenge)
        return direction_array

    def residual_update(self, x, y, s, z, k, t):
        sk = np.concatenate([self.zero_n1, self.zero_p1, np.vstack(s), [[k]]])
        xyzt = np.concatenate([x, y, np.vstack(z), [[t]]])

        return sk - np.dot(self.KKT_Matrix, xyzt)
