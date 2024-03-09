import numpy as np
import math
import function_algorithm1
import para


def fairness(M, N, C_u, Band, pw, N0, h_u, p_id, p_com, p_loc, lamda_e, lamda_t, Din, Dout, omg, f_loc, f_c, prob):
    a = np.ones([M]).reshape(-1, 1) / M
    C, D = function_algorithm1.algorithm1(M, N, C_u, Band, a, pw, N0, h_u, p_id, p_com, p_loc, lamda_e, lamda_t,
                                          Din, Dout, omg, f_loc, f_c, prob)
    r = a * Band * np.log2(1 + pw * h_u * math.pow(10, -6) / (math.pow(10, N0 / 10) / 1000))
    A = prob * (lamda_e * p_com + lamda_t) * Din / r
    B = prob * (lamda_e * p_loc + lamda_t) * Din * omg / f_loc
    F = prob * (lamda_e * p_id + lamda_t) * Din * omg / f_c
    G = prob * (lamda_e * p_com + lamda_t) * Dout / r
    H = F + G

    Cost_temp = -A * C + (A + B - F - G) * D + H
    Cost_u = np.sum(Cost_temp, axis=1).reshape(-1, 1)

    Cost_min = np.min(Cost_u)
    Cost_max = np.max(Cost_u)
    print(C)
    print(D)

    return Cost_min, Cost_max, Cost_u


multiple = 1
M = 5 * multiple
N = 10
C_u = 4
Band = 40 * math.pow(10, 6)
pw = 0.1
N0 = -174
h_u = np.ones(M).reshape(-1, 1) * 2
p_id = np.array([0.1, 0.3, 0.5, 0.7, 0.9]).reshape(-1, 1) * 0.008 + 0.001
p_com = np.array([0.1, 0.3, 0.5, 0.7, 0.9]).reshape(-1, 1) * 0.08 + 0.01
p_loc = np.array([0.1, 0.3, 0.5, 0.7, 0.9]).reshape(-1, 1) * 0.4 + 0.1
p_id = np.concatenate([p_id]*multiple)
p_com = np.concatenate([p_com]*multiple)
p_loc = np.concatenate([p_loc]*multiple)
# Coste = 0.0014278790842297854
# Costu = 0.0442443658711503
Coste = 1
Costu = 1
lamda_e = 0.2 / Coste
lamda_t = 0.8 / Costu
Din = 3 * math.pow(10, 6)
Dout = (np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]).reshape(1, -1) * 2 + 6) * math.pow(10, 6)
omg = np.array([0.05, 0.15, 0.95, 0.65, 0.45, 0.55, 0.35, 0.75, 0.85, 0.25]).reshape(1, -1) * 10 + 10
f_loc = np.array([0.75, 0.5, 1.5, 1.25, 1]).reshape(-1, 1) * np.power(10, 9)
f_loc = np.concatenate([f_loc]*multiple)
f_c = 2 * math.pow(10, 9)

real_prob = para.real_prob
real_prob = np.concatenate([real_prob]*multiple)

Cost_min, Cost_max, Cost_u = fairness(M, N, C_u, Band, pw, N0, h_u, p_id, p_com, p_loc, lamda_e, lamda_t, Din, Dout, omg, f_loc, f_c, prob=real_prob)
print(f"Cost_min = {Cost_min}")
print(f"Cost_max = {Cost_max}")
# print(f"Cost_u = {Cost_u}")
