import numpy as np
from kkt import KKT
from cone import CONE


class PWORK():
    def __init__(self, c, A, b, G, h, Cone_Dim):
        self.c = c.copy()
        self.A = A.copy()
        self.b = b.copy()
        self.G = G.copy()
        self.h = h.copy()
        self.Cone_Dim = Cone_Dim.copy()

        self.p = b.shape[0]  # 等式约束个数
        self.m = h.shape[0]  # 不等式约束个数
        self.n = c.shape[0]  # x维数

        self.fengge = np.add.accumulate(self.Cone_Dim)
        self.Cone_Dim_bar = self.Cone_Dim.copy()
        self.G_bar, self.h_bar = self.Gh_fengge()
        self.fengge_bar = np.add.accumulate(self.Cone_Dim_bar)

        self.kkt = KKT(c, A, b, self.G, self.h, self.p, self.m, self.n, self.fengge_bar)
        self.cone = CONE(Cone_Dim)

        self.kappa = 1
        self.tau = 1

        # self.x = np.random.randint(10, size=[self.n, 1])
        # self.y = np.random.randint(10, size=[self.p, 1])
        self.x = np.ones([self.n, 1])
        self.y = np.ones([self.p, 1])

        self.s = self.cone.init_value
        self.z = self.cone.init_value

    def solve(self):
        for i in range(12):
            s_value = np.vstack(self.s)
            z_value = np.vstack(self.z)

            # 第一步，compute residuals and evaluate stopping criteria
            r_x, r_y, r_z, r_tau = self.kkt.residual_update(self.x, self.y, s_value, z_value, self.kappa, self.tau)
            # 检查退出条件

            print(r_x, r_y, r_z, r_tau)
            # 第二步，compute scaling matrix W
            W, W_2 = self.cone.W_square(self.z, self.s)
            W_inv = self.cone.inv_t(W)
            W_matrix = self.cone.pinjie(W, self.Cone_Dim)
            W_2_matrix = self.cone.pinjie(W_2, self.Cone_Dim_bar)

            lambda_v = self.cone.dot(W, self.z)
            miu = (self.cone.sT_z(self.s, self.z) + self.kappa * self.tau) / (self.cone.Cone_num + 1)

            # 第三步，compute affine scaling direction
            dx = -1 * r_x
            dy = -1 * r_y
            dz = -1 * r_z
            d_tau = -1 * r_tau

            lambda_lambda = self.cone.Vector_Product(lambda_v, lambda_v)
            ds = -1 * np.vstack(lambda_lambda)
            d_kappa = np.diag([-1 * self.kappa * self.tau])

            bx = -1 * dx
            by = dy
            # bz = dz - np.dot(W_matrix, np.vstack(self.cone.divide(lambda_v, self.cone.fenge(ds))))
            bz = dz - s_value
            b_tau = -1 * d_tau
            bs = -1 * ds
            lambda_bs = np.vstack(self.cone.divide(lambda_v, self.cone.fenge(bs)))

            b_kappa = -1 * d_kappa
            x_direction_affline, y_direction_affline, z_direction_affline, tau_direction_affline, s_direction_affline, kappa_direction_affline = self.kkt.solve(
                0, W_matrix, W_2_matrix, bx, by, bz, lambda_bs, b_tau, b_kappa, self.kappa, self.tau)

            z_direction_affline_cone = self.cone.fenge(z_direction_affline)
            s_direction_affline_cone = self.cone.fenge(s_direction_affline)
            tau_direction_affline = tau_direction_affline[0][0]
            kappa_direction_affline = kappa_direction_affline[0][0]

            # 第四步，select barrier parameter
            distance_z = self.cone.Distance_to_boundary(self.z, z_direction_affline_cone)
            distance_s = self.cone.Distance_to_boundary(self.s, s_direction_affline_cone)
            if tau_direction_affline >= 0:
                distance_tau = float('inf')
            else:
                distance_tau = -1 * self.tau / tau_direction_affline

            if kappa_direction_affline >= 0:
                distance_kappa = float('inf')
            else:
                distance_kappa = -1 * self.kappa / kappa_direction_affline
            alpha = min([min([distance_z, distance_s, distance_tau, distance_kappa]), 1])
            sigma = (1 - alpha) ** 3

            # 第五步 compute search direction
            dx = -1 * (1 - sigma) * r_x
            dy = -1 * (1 - sigma) * r_y
            dz = -1 * (1 - sigma) * r_z
            d_tau = -1 * (1 - sigma) * r_tau
            ds = -1 * np.vstack(lambda_lambda) + sigma * miu * np.vstack(self.cone.init_value) - np.vstack(
                self.cone.Vector_Product(self.cone.dot(W_inv, s_direction_affline_cone),
                                         self.cone.dot(W, z_direction_affline_cone)))
            d_kappa = np.diag(
                [-1 * self.kappa * self.tau + sigma * miu - tau_direction_affline * kappa_direction_affline])

            bx = -1 * dx
            by = 1 * dy
            bz = 1 * dz - np.vstack(self.add_two(self.cone.dot(W, self.cone.divide(lambda_v, self.cone.fenge(ds)))))
            b_tau = -1 * d_tau
            bs = -1 * ds
            lambda_bs = np.vstack(self.cone.divide(lambda_v, self.cone.fenge(bs)))
            b_kappa = -1 * d_kappa

            x_direction, y_direction, z_direction, tau_direction, s_direction, kappa_direction = self.kkt.solve(
                1, W_matrix, W_2_matrix, bx, by, bz, lambda_bs, b_tau, b_kappa, self.kappa, self.tau)
            z_direction_cone = self.cone.fenge(z_direction)
            s_direction_cone = self.cone.fenge(s_direction)
            tau_direction = tau_direction[0][0]
            kappa_direction = kappa_direction[0][0]

            # 第六步 update iterates
            distance_z = self.cone.Distance_to_boundary(self.z, z_direction_cone)
            distance_s = self.cone.Distance_to_boundary(self.s, s_direction_cone)

            if tau_direction >= 0:
                distance_t = float('inf')
            else:
                distance_t = -1 * self.tau / tau_direction

            if kappa_direction >= 0:
                distance_k = float('inf')
            else:
                distance_k = -1 * self.kappa / kappa_direction
            alpha = 0.99 * min([min([distance_z, distance_s, distance_t, distance_k]), 1])

            self.x = self.x + alpha * x_direction
            self.y = self.y + alpha * y_direction
            self.z = self.cone.fenge(np.vstack(self.z) + alpha * z_direction)
            self.tau = self.tau + alpha * tau_direction
            self.s = self.cone.fenge(np.vstack(self.s) + alpha * s_direction)
            self.kappa = self.kappa + alpha * kappa_direction

            print(i, np.dot(self.c.transpose(), self.x)[0][0] / self.tau, self.kappa, self.tau)

    def Gh_fengge(self):
        G_array = np.vsplit(self.G, self.fengge)
        h_array = np.vsplit(self.h, self.fengge)

        zero_2n = np.zeros([2, self.n])
        zero_21 = np.zeros([2, 1])
        G_bar = np.zeros([0, self.n])
        h_bar = np.zeros([0, 1])
        for i in range(len(self.Cone_Dim)):
            G_bar = np.vstack([G_bar, G_array[i]])
            h_bar = np.vstack([h_bar, h_array[i]])
            if self.Cone_Dim[i] == 1:
                continue
            else:
                self.Cone_Dim_bar[i] = self.Cone_Dim_bar[i] + 2
                G_bar = np.vstack([G_bar, zero_2n])
                h_bar = np.vstack([h_bar, zero_21])
        return [G_bar, h_bar]

    def add_two(self, s):
        return_value = s.copy()
        for i in range(len(self.Cone_Dim)):
            if self.Cone_Dim[i] == 1:
                continue
            else:
                return_value[i] = np.vstack([return_value[i], np.zeros([2, 1])])
        return return_value

    def sub_two(self, s):
        return_value = s.copy()
        for i in range(len(self.Cone_Dim)):
            if self.Cone_Dim[i] == 1:
                continue
            else:
                return_value[i] = return_value[i][:-2]
        return return_value
