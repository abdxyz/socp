import numpy as np


class KKT():
    def __init__(self, c, A, b, G, h, p, m, n):
        self.c = c.copy()
        self.A = A.copy()
        self.b = b.copy()
        self.G = G.copy()
        self.h = h.copy()

        self.cbh = np.vstack([-1 * self.c, self.b, self.h])

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

        self.Residual_Matrix = self.residual_matrix(c, A, b, G, h)
        self.Direction_Matrix = self.direction_matrix(c, A, b, G, h)
        self.delta_x1, self.delta_y1, self.delta_z1 = None, None, None

        self.equ_fenge = [n, p, m, 1, m, 1]
        self.fenge = np.add.accumulate(self.equ_fenge)

    def residual_matrix(self, c, A, b, G, h):
        hang_0 = np.concatenate((self.zero_nn, self.A.transpose(), self.G.transpose(), self.c), axis=1)
        hang_1 = np.concatenate((-1 * self.A, self.zero_pp, self.zero_mp.transpose(), b), axis=1)
        hang_2 = np.concatenate((-1 * self.G, self.zero_mp, self.zero_mm, self.h), axis=1)
        hang_3 = np.concatenate(
            (-1 * self.c.transpose(), -1 * self.b.transpose(), -1 * self.h.transpose(), self.zero_11), axis=1)
        return np.concatenate((hang_0, hang_1, hang_2, hang_3), axis=0)

    def direction_matrix(self, c, A, b, G, h):
        hang_0 = np.concatenate((self.zero_nn, self.A.transpose(), self.G.transpose()), axis=1)
        hang_1 = np.concatenate((self.A, self.zero_pp, self.zero_mp.transpose()), axis=1)
        hang_2 = np.concatenate((self.G, self.zero_mp, self.zero_mm), axis=1)
        return np.concatenate((hang_0, hang_1, hang_2), axis=0)

    def solve(self, direction_type, W, bx, by, bz, lambda_bs, b_tau, b_kappa, kappa, tau):
        minus_w2 = -1 * np.dot(W, W)
        b_xyz = np.vstack([bx, by, bz])
        if direction_type == 0:
            self.Direction_Matrix[self.fenge[1]:self.fenge[2], self.fenge[1]:self.fenge[2]] = minus_w2
            self.delta_x1, self.delta_y1, self.delta_z1 = np.vsplit(np.linalg.solve(self.Direction_Matrix, self.cbh),
                                                                    self.fenge[0:2])
        delta_x2, delta_y2, delta_z2 = np.vsplit(np.linalg.solve(self.Direction_Matrix, b_xyz),
                                                 self.fenge[0:2])

        delta_tau = (b_tau - b_kappa / tau + np.dot(self.c.transpose(), delta_x2)[0] + np.dot(self.b.transpose(),
                                                                                              delta_y2)[0] + np.dot(
            self.h.transpose(), delta_z2)) / (kappa / tau - np.dot(self.c.transpose(), self.delta_x1)[0] -
                                              np.dot(self.b.transpose(), self.delta_y1)[0] -
                                              np.dot(self.h.transpose(), self.delta_z1)[0])
        delta_x = delta_x2 + delta_tau * self.delta_x1
        delta_y = delta_y2 + delta_tau * self.delta_y1
        delta_z = delta_z2 + delta_tau * self.delta_z1
        delta_s = -1 * np.dot(W, lambda_bs + np.dot(W, delta_z))
        delta_kappa = -(b_kappa + kappa * delta_tau) / tau
        return delta_x, delta_y, delta_z, delta_tau, delta_s, delta_kappa

    def residual_update(self, x, y, s, z, k, t):
        sk = np.concatenate([self.zero_n1, self.zero_p1, np.vstack(s), [[k]]])
        xyzt = np.concatenate([x, y, np.vstack(z), [[t]]])

        residual_vec = sk - np.dot(self.Residual_Matrix, xyzt)
        return np.vsplit(residual_vec, self.fenge[0:3])
