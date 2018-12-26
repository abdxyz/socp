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

        self.kkt = KKT(c, A, b, G, h, self.p, self.m, self.n)
        self.cone = CONE(Cone_Dim)

        self.k = 1
        self.t = 1
        self.x = np.random.randint(10, size=[self.n, 1])
        self.y = np.random.randint(10, size=[self.p, 1])
        self.s = self.cone.init_value
        self.z = self.cone.init_value

    def solve(self):
        for i in range(40):
            # 第一步，compute residuals and evaluate stopping criteria
            residual = self.kkt.residual_update(self.x, self.y, self.s, self.z, self.k, self.t)
            # 检查退出条件

            # 第二步，compute scaling matrix W
            W = self.cone.W(self.z, self.s)
            W_inv = self.cone.inv_t(W)

            lambda_v = self.cone.dot(W, self.z)

            lambda_v_new = self.cone.dot(W_inv, self.s)

            miu = (self.cone.sT_z(self.s, self.z) + self.k * self.t) / (self.cone.Cone_num + 1)

            # 第三步，compute affine scaling direction
            lambdalambda = self.cone.Vector_Product(lambda_v, lambda_v)
            minus_lambdalambda = -1 * np.vstack(lambdalambda)

            minus_residual = -1 * residual
            minus_kt = np.diag([-1 * self.k * self.t])
            equation_right = np.vstack((minus_residual, minus_lambdalambda, minus_kt))

            L_lambda = self.cone.L_x(lambda_v)

            W1 = self.cone.pinjie(self.cone.dot(L_lambda, W))
            W2 = self.cone.pinjie(self.cone.dot(L_lambda, W_inv))

            afflinedirection_array = self.kkt.solve(equation_right, W1, W2, self.k, self.t)

            x_direction_affline = afflinedirection_array[0]
            y_direction_affline = afflinedirection_array[1]
            z_direction_affline = self.cone.fenge(afflinedirection_array[2])
            t_direction_affline = afflinedirection_array[3][0][0]
            s_direction_affline = self.cone.fenge(afflinedirection_array[4])
            k_direction_affline = afflinedirection_array[5][0][0]

            # 第四步，select barrier parameter
            distance_z = self.cone.Distance_to_boundary(self.z, z_direction_affline)
            distance_s = self.cone.Distance_to_boundary(self.s, s_direction_affline)
            if t_direction_affline >= 0:
                distance_t = float('inf')
            else:
                distance_t = -1 * self.t / t_direction_affline

            if k_direction_affline >= 0:
                distance_k = float('inf')
            else:
                distance_k = -1 * self.k / k_direction_affline
            alpha = min([min([distance_z, distance_s, distance_t, distance_k]), 1])
            sigma = (1 - alpha) ** 3

            # 第五步 compute search direction
            minus_residual_sigma = minus_residual * (1 - sigma)

            sigma_miu_e = sigma * miu * np.vstack(self.cone.init_value)
            wswz = np.vstack(
                self.cone.Vector_Product(self.cone.dot(W_inv, s_direction_affline),
                                         self.cone.dot(W, z_direction_affline)))

            kt = np.diag([sigma * miu - self.k * self.t - k_direction_affline * t_direction_affline])

            equation_right = np.vstack((minus_residual_sigma, minus_lambdalambda + sigma_miu_e - wswz, kt))

            searchdirection_array = self.kkt.solve(equation_right, W1, W2, self.k, self.t)

            x_direction = searchdirection_array[0]
            y_direction = searchdirection_array[1]
            z_direction = self.cone.fenge(searchdirection_array[2])
            t_direction = searchdirection_array[3][0][0]
            s_direction = self.cone.fenge(searchdirection_array[4])
            k_direction = searchdirection_array[5][0][0]

            # 第六步 update iterates
            distance_z = self.cone.Distance_to_boundary(self.z, z_direction)
            distance_s = self.cone.Distance_to_boundary(self.s, s_direction)

            if t_direction >= 0:
                distance_t = float('inf')
            else:
                distance_t = -1 * self.t / t_direction

            if k_direction >= 0:
                distance_k = float('inf')
            else:
                distance_k = -1 * self.k / k_direction
            alpha = 0.99 * min([min([distance_z, distance_s, distance_t, distance_k]), 1])
            # print(self.cone.Vector_Product(self.s, self.z))
            self.x = self.x + alpha * x_direction
            self.y = self.y + alpha * y_direction
            self.z = self.cone.fenge(np.vstack(self.z) + alpha * searchdirection_array[2])
            self.t = self.t + alpha * t_direction
            self.s = self.cone.fenge(np.vstack(self.s) + alpha * searchdirection_array[4])
            self.k = self.k + alpha * k_direction

            # print(self.cone.check(self.z))
            # print(self.cone.check(self.s))
            # print(self.c)
            print(i, np.dot(np.vstack(self.s).transpose(), np.vstack(self.z))[0][0], np.linalg.norm(residual),
                  np.dot(self.c.transpose(), self.x)[0][0] / self.t,
                  np.dot(self.h.transpose(), np.vstack(self.z)) / self.t)
            print(self.k, self.t)
