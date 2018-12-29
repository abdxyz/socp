import numpy as np


class CONE():
    def __init__(self, Cone_Dim):
        self.Cone_Dim = Cone_Dim
        self.Dim_Sum = sum(Cone_Dim)
        self.Cone_num = len(Cone_Dim)
        self.j_array = self.J_Array()
        self.init_value = self.init_cone()

        self.fenge_arr = np.add.accumulate(self.Cone_Dim)

    def init_cone(self):
        J_array = list()
        one = [1]
        zero = [0]
        for item in self.Cone_Dim:
            zero_arr = zero * (item - 1)
            J_array.append(np.array(one + zero_arr).reshape(-1, 1))
        return J_array

    def x_inv(self, x):
        x_inv = list()
        for i in range(self.Cone_num):
            x_inv.append(np.dot(self.j_array[i], x[i]) / np.dot(np.dot(x[i].transpose(), self.j_array[i]), x[i])[0][0])
        return x_inv

    def x_square(self, x):
        x_square = list()
        for i in range(self.Cone_num):
            square = x[i].copy()
            length = (np.dot(np.dot(x[i].transpose(), self.j_array[i]), x[i])[0][0]) ** (1 / 2)
            constant = 1 / (2 ** (1 / 2) * (x[i][0][0] + length) ** (1 / 2))
            square[0][0] = x[i][0][0] + length
            square = square * constant
            x_square.append(square)
        return x_square

    def L_x(self, x):
        l_x = list()
        for i in range(self.Cone_num):
            x0 = x[i][0, :].reshape([1, 1])
            x1 = x[i][1:, :]

            heng_0 = np.hstack([x0, x1.transpose()])
            heng_1 = np.hstack([x1, x0 * np.eye(self.Cone_Dim[i] - 1)])

            l_x.append(np.vstack([heng_0, heng_1]))
        return l_x

    def P_x(self, x):
        l_x = self.L_x(x)

        x_2 = self.Vector_Product(x, x)
        x_2_matrix = self.L_x(x_2)

        p_x = list()
        for i in range(self.Cone_num):
            p_x.append(2 * np.dot(l_x[i], l_x[i]) - x_2_matrix[i])

        return p_x

    def W(self, z, s):
        z1_2 = self.x_square(z)
        zminus1_2 = self.x_inv(z1_2)
        Pminus1_2 = self.P_x(zminus1_2)

        p1_2 = self.P_x(z1_2)
        p1_2s1_2 = self.x_square(self.dot(p1_2, s))
        omega = self.dot(Pminus1_2, p1_2s1_2)
        omega_tidal = self.x_square(omega)
        return self.P_x(omega_tidal)

    def W_square(self, z, s):
        W_list = list()
        for i in range(self.Cone_num):
            if self.Cone_Dim[i] == 1:
                W_list.append(np.diag([s[i][0][0] / z[i][0][0]]))
            else:
                z_i = z[i]
                s_i = s[i]
                length_zi = (np.dot(np.dot(z_i.transpose(), self.j_array[i]), z_i)) ** (1 / 2)
                length_si = (np.dot(np.dot(s_i.transpose(), self.j_array[i]), s_i)) ** (1 / 2)
                zi_bar = z_i / length_zi
                si_bar = s_i / length_si

                gamma = ((1 + np.dot(zi_bar.transpose(), si_bar)[0]) / 2) ** (1 / 2)
                w_bar = (si_bar + np.dot(self.j_array[i], zi_bar)) / (2 * gamma)
                eta_square = length_si / length_zi

                w1_length = (np.linalg.norm(w_bar[1:])) ** 2

                c = 1 + w_bar[0][0] + w1_length / (1 + w_bar[0][0])
                d = 1 + 2 / (1 + w_bar[0][0]) + w1_length / (1 + w_bar[0][0]) ** 2

                a = ((w_bar[0][0]) ** 2 + w1_length - (c ** 2 * w1_length) / (1 + d * w1_length)) / 2
                u_0 = ((w_bar[0][0]) ** 2 + w1_length - a) ** (1 / 2)
                u_1 = c / u_0
                v_1 = (c ** 2 / u_0 ** 2 - d) ** (1 / 2)

                D = np.diag([a] + [1] * (self.Cone_Dim[i] - 1))
                u = np.vstack([np.diag([u_0]), u_1 * w_bar[1:]])
                v = np.vstack([np.zeros([1, 1]), v_1 * w_bar[1:]])

                V = eta_square * (D + np.dot(u, u.transpose()) - np.dot(v, v.transpose()))
                W_list.append(V)
        return W_list

    def Vector_Product(self, x, y):
        L_x = list()
        for i in range(self.Cone_num):
            x0 = x[i][0, :]
            x1 = x[i][1:, :]
            y0 = y[i][0, :]
            y1 = y[i][1:, :]

            z0 = np.dot(x[i].transpose(), y[i])
            z1 = x0 * y1 + x1 * y0

            L_x.append(np.vstack([z0, z1]))
        return L_x

    def Distance_to_boundary(self, x, y):
        distance_list = list()

        x_inv_v = self.x_inv(x)
        x_inv_12 = self.x_square(x_inv_v)
        Px_12_y = self.dot(self.P_x(x_inv_12), y)

        for i in range(self.Cone_num):
            if self.Cone_Dim[i] >= 2:
                length = y[i][0][0] - np.linalg.norm(y[i][1:])
            else:
                length = y[i][0][0]

            if length >= 0:
                distance_list.append(float("inf"))
            else:
                lambda_min = -1 * (Px_12_y[i][0][0] - np.linalg.norm(Px_12_y[i][1:]))
                lambda_min = 1 / lambda_min
                distance_list.append(lambda_min)
        return min(distance_list)

    def sT_z(self, s, z):
        result = 0
        for i in range(self.Cone_num):
            temp = np.dot(s[i].transpose(), z[i])[0][0]
            result = result + temp
        return result

    def J_Array(self):
        J_array = list()
        one = [1]
        minus = [-1]
        for item in self.Cone_Dim:
            minus_arr = minus * (item - 1)
            J_array.append(np.diag(one + minus_arr))
        return J_array

    def dot(self, P, s):
        x = list()
        for i in range(self.Cone_num):
            dot_i = np.dot(P[i], s[i])
            x.append(dot_i)
        return x

    def inv_t(self, W):
        x = list()
        for i in range(self.Cone_num):
            inv_t_i = np.linalg.inv(W[i]).transpose()
            x.append(inv_t_i)
        return x

    def t(self, s):
        x = list()
        for i in range(self.Cone_num):
            t_i = s[i].transpose()
            x.append(t_i)
        return x

    def pinjie(self, W):
        x = np.zeros([self.Dim_Sum, self.Dim_Sum])
        begin = 0
        for i in range(self.Cone_num):
            x[begin:begin + self.Cone_Dim[i], begin:begin + self.Cone_Dim[i]] = W[i]
            begin = begin + self.Cone_Dim[i]
        return x

    def fenge(self, s):
        return np.vsplit(s, self.fenge_arr)

    def check(self, y):
        for i in range(self.Cone_num):
            if self.Cone_Dim[i] >= 2:
                length = y[i][0][0] - np.linalg.norm(y[i][1:])
            else:
                length = y[i][0][0]

            if length < 0:
                return 0
        return 1

    def divide(self, u, w):
        divide_value = list()
        for i in range(self.Cone_num):
            if self.Cone_Dim[i] >= 2:
                Q = np.dot(u[i].transpose(), np.dot(self.j_array[i], u[i]))[0][0]
                v = np.dot(u[i][1:].transpose(), w[i][1:])[0][0]
                first = np.diag([u[i][0][0] * w[i][0][0] - v])
                second = (v / u[i][0][0] - w[i][0][0]) * u[i][1:] + Q / u[i][0][0] * w[i][1:]
                divide_value.append(np.vstack([first, second]) / Q)
            else:
                divide_value.append(np.diag([w[i][0][0] / u[i][0][0]]))
        return divide_value


# cone = CONE([3, 2, 1])
# cone_0 = np.array([100, 0, 0]).reshape([3, 1])
# cone_1 = np.array([2.0, 1.0]).reshape([2, 1])
# cone_2 = np.array([1]).reshape([1, 1])
# s = [cone_0, cone_1, cone_2]
#
# cone_0 = np.array([30, 0, 0]).reshape([3, 1])
# cone_1 = np.array([5.0, 2.0]).reshape([2, 1])
# cone_2 = np.array([3]).reshape([1, 1])
# z = [cone_0, cone_1, cone_2]
# W = cone.W(z, s)
# print(np.dot(cone.pinjie(W), cone.pinjie(W)))
# print(cone.W_square(z, s))
