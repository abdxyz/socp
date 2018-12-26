import numpy as np
import math


def Px(x, j):
    return np.dot(x, x.transpose()) - np.dot(np.dot(x.transpose(), j), x) / 2 * j


def x1_2(x, j):
    length = np.dot(np.dot(x.transpose(), j), x)
    x0 = x[0]
    constant = 1 / (2 ** (1 / 4) * (x0 + length ** (1 / 2)) ** (1 / 2))
    x1 = x[1:]
    lie = np.vstack((x0 + length ** (1 / 2), x1))
    return constant * lie


def x_inv(x, j):
    return 2 / np.dot(np.dot(x.transpose(), j), x) * np.dot(j, x)


def Lx(x, Kfengge):
    xarray = np.vsplit(x, Kfengge[:-1])

    Lx_v = np.zeros([x.shape[0], x.shape[0]])

    begin = 0
    for i in range(Kfenge.size):
        x0 = xarray[i][0][0]
        Lx_temp = np.diag([x0] * xarray[i].shape[0])
        x1 = xarray[i][1:].flatten()
        Lx_temp[0, 1:] = x1.transpose()
        Lx_temp[1:, 0] = x1
        Lx_v[begin:Kfenge[i], begin:Kfenge[i]] = Lx_temp
        begin = Kfenge[i]
    return Lx_v / math.sqrt(2)


def omega(z, s, j):
    z1_2 = x1_2(z, j)
    zminus1_2 = x_inv(z1_2, j)
    Pminus1_2 = Px(zminus1_2, j)
    p1_2 = Px(z1_2, j)
    p1_2s1_2 = x1_2(np.dot(p1_2, s), j)
    return np.dot(Pminus1_2, p1_2s1_2)


def W(z, s, j):
    omega_v = omega(z, s, j)
    omega_tidel = x1_2(omega_v, j)
    return Px(omega_tidel, j)


def vector_product(x_src, y_src, Kfenge):
    xarray = np.vsplit(x_src, Kfenge[:-1])
    yarray = np.vsplit(y_src, Kfenge[:-1])

    vec_pro = np.zeros([h_num, 1])
    begin = 0
    for i in range(Kfenge.size):
        x = xarray[i]
        y = yarray[i]
        first = np.dot(x.transpose(), y)
        xafter = x[1:]
        yafter = y[1:]
        after = y[0] * xafter + x[0] * yafter
        pro_temp = np.vstack((first, after)) / math.sqrt(2)
        vec_pro[begin:Kfenge[i]] = pro_temp
        begin = Kfenge[i]
    return vec_pro


def sigma_xy(x_src, y_src, jarray, Kfenge):
    xarray = np.vsplit(x_src, Kfenge[:-1])
    yarray = np.vsplit(y_src, Kfenge[:-1])

    distance = np.zeros([Kfenge.size])
    for i in range(Kfenge.size):
        x = xarray[i]
        y = yarray[i]
        j = jarray[i]

        # print(x)
        # print(y)
        # print(j)
        if y[0][0] > np.linalg.norm(y[1:, 0]):
            lambda_min = float("inf")
        else:
            x_inv_v = x_inv(x, j)
            x_inv_12 = x1_2(x_inv_v, j)
            Px_12_y = np.dot(Px(x_inv_12, j), y)
            lambda_min = -1 * (Px_12_y[0][0] - np.linalg.norm(Px_12_y[1:])) / math.sqrt(2)
            lambda_min = 1 / lambda_min
        distance[i] = lambda_min
    return distance


x_dim = 5  # x的维度
h_num = 10  # 锥的维度
K = np.array([3, 3, 4])
Kfenge = np.array([3, 6, 10])

j0 = np.diag([1, -1, -1])
j1 = np.diag([1, -1, -1])
j2 = np.diag([1, -1, -1, -1])
J = [j0, j1, j2]

unit = np.zeros([h_num, 1])
begin = 0
for i in range(Kfenge.size):
    unit[begin] = math.sqrt(2)
    begin = Kfenge[i]

# A = np.random.randint(10, size=[h_num, x_dim])
A = np.array([[5, 6, 1, 0, 3],
              [2, 9, 2, 9, 8],
              [6, 0, 8, 6, 9],
              [7, 4, 3, 4, 6],
              [6, 9, 1, 4, 5],
              [0, 1, 1, 9, 1],
              [8, 6, 3, 0, 6],
              [6, 7, 7, 2, 7],
              [8, 3, 9, 9, 5],
              [7, 0, 5, 3, 8]])
# b = np.random.randint(10, size=[h_num, 1])
b = np.array([121, 146, 164, 84, 64, 108, 127, 93, 155, 141]).reshape([10, 1])
# c = np.random.randint(10, size=[x_dim, 1])
c = np.array([33, 88, 89, 78, 58]).reshape([5, 1])
# print(np.linalg.matrix_rank(A))

s = np.zeros([h_num, 1])
s[0] = 3
s[1] = 2
s[3] = 3
s[4] = 1
s[6] = 5
s[7] = 2

z = np.zeros([h_num, 1])
z[0] = 4
z[1] = 2.1
z[3] = 5
z[4] = 2
z[6] = 1
z[7] = 0.2

# print(zarray)

x = np.random.randint(10, size=[x_dim, 1])
# mu = 100
# sigma = 1
for j in range(20):
    zarray = np.vsplit(z, Kfenge[:-1])
    sarray = np.vsplit(s, Kfenge[:-1])

    # 第一步，compute residuals and evaluate stopping criteria
    zero_xdim = np.zeros((x_dim, 1))
    szero = np.vstack((zero_xdim, s))
    cb = np.vstack((c, b))

    zero_xdimxdim = np.zeros((x_dim, x_dim))
    zero_hnumhnum = np.zeros((h_num, h_num))
    hengpinjie1 = np.hstack((zero_xdimxdim, A.transpose()))
    hengpinjie2 = np.hstack((-1 * A, zero_hnumhnum))
    shupinjie = np.vstack((hengpinjie1, hengpinjie2))

    xz = np.vstack((x, z))
    residual = szero - np.dot(shupinjie, xz) - cb

    # 第二步，compute scaling matrix W
    W_v = np.zeros([h_num, h_num])
    begin = 0
    for i in range(K.shape[0]):
        w = W(zarray[i], sarray[i], J[i])
        W_v[begin:Kfenge[i], begin:Kfenge[i]] = w
        begin = Kfenge[i]
    W_v_inv = np.linalg.inv(W_v).transpose()

    lambda_v = np.dot(W_v, z)
    mu = np.dot(lambda_v.transpose(), lambda_v)[0, 0]

    # 第三步，compute affine scaling direction
    # x = np.array([-10, 19, 2]).reshape((3, 1))
    # y = np.array([4, 1, 10]).reshape((3, 1))
    # print(vector_product(x, y))
    # print(Lx(x))

    minus_lambdalambda = -1 * vector_product(lambda_v, lambda_v, Kfenge)
    minus_residual = -1 * residual
    equation_right = np.vstack((minus_residual, minus_lambdalambda))

    L_lambda = Lx(lambda_v, Kfenge)

    W1 = np.dot(L_lambda, W_v)
    W2 = np.dot(L_lambda, W_v_inv)
    hang_0 = np.concatenate((zero_xdimxdim, -1 * A.transpose(), np.zeros([x_dim, h_num])), axis=1)
    hang_1 = np.concatenate((A, zero_hnumhnum, np.eye(h_num)), axis=1)
    hang_2 = np.concatenate((np.zeros([h_num, x_dim]), W1, W2), axis=1)
    matrix = np.concatenate((hang_0, hang_1, hang_2), axis=0)

    afflinedirection = np.linalg.solve(matrix, equation_right)
    afflinedirection_array = np.vsplit(afflinedirection, [x_dim, x_dim + h_num])

    # 第四步，select barrier parameter
    z_direction_affline = afflinedirection_array[1]
    s_direction_affline = afflinedirection_array[2]

    alpha_p = sigma_xy(s, s_direction_affline, J, Kfenge)
    alpha_d = sigma_xy(z, z_direction_affline, J, Kfenge)
    alpha_p_min = min(min(alpha_p), 1)
    alpha_d_min = min(min(alpha_d), 1)

    delta = 3
    sigma = (np.dot((s + alpha_p_min * s_direction_affline).transpose(), (z + alpha_d_min * z_direction_affline))[0][
                 0] /
             np.dot(s.transpose(), z)[0][0]) ** delta

    # 第五步 compute search direction
    # Mehrotra correction
    # sigma_mu_unit = sigma * mu * unit - vector_product(np.dot(W_v_inv, s_direction_affline),
    #                                                   np.dot(W_v, z_direction_affline),
    #                                                   Kfenge)
    sigma_mu_unit = sigma * mu * unit
    equation_right = np.vstack((minus_residual, minus_lambdalambda + sigma_mu_unit))
    search_direction = np.linalg.solve(matrix, equation_right)
    searchdirection_array = np.vsplit(search_direction, [x_dim, x_dim + h_num])

    # 第六步 update iterates
    x_direction = searchdirection_array[0]
    z_direction = searchdirection_array[1]
    s_direction = searchdirection_array[2]

    alpha_p = sigma_xy(s, s_direction, J, Kfenge)
    alpha_d = sigma_xy(z, z_direction, J, Kfenge)

    alpha_p_min = min(min(alpha_p) * 0.99, 1)
    alpha_d_min = min(min(alpha_d) * 0.99, 1)

    x = x + alpha_p_min * x_direction
    s = s + alpha_p_min * s_direction
    z = z + alpha_d_min * z_direction

    print(j, np.dot(s.transpose(), z)[0][0], np.linalg.norm(residual), np.dot(c.transpose(), x)[0][0])