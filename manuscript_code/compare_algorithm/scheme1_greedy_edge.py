import numpy as np
import math
import function_algorithm2
import para
import T_E

def scheme1(M, N, C_u, Band, pw, N0, h_u, p_id, p_com, p_loc, lamda_e, lamda_t, Din, Dout, omg, f_loc, f_c, prob_change,
            prob_fixed):
    a = np.ones([M]).reshape(-1, 1) / M
    Cost_max = math.pow(10, 10)
    Cost_opt = -1
    iteration1 = 0

    while np.fabs(Cost_max - Cost_opt) != math.pow(10, -12):
        iteration1 += 1
        temp = Cost_max - Cost_opt
        print(temp)
        Cost_opt = Cost_max
        C = np.zeros([M, N])
        D = np.zeros([M, N])

        a, Cost_max, Cost_max_index = function_algorithm2.algorithm2(M, N, C_u, Band, pw, N0, h_u, p_id, p_com, p_loc, lamda_e, lamda_t,
                                                     Din, Dout, omg, f_loc, f_c, prob_fixed, C, D)
        if np.fabs(Cost_max - Cost_opt) <= math.pow(10, -2):
            print(Cost_max - Cost_opt)
            return C, D, a, Cost_max, Cost_max_index


multiple = 1
M = 5 * multiple
N = 10
C_u = 4
Band = 30 * math.pow(10, 6)
pw = 0.1
N0 = -174
h_u = np.ones(M).reshape(-1, 1) * 2
p_id = np.array([0.1, 0.3, 0.5, 0.7, 0.9]).reshape(-1, 1) * 0.008 + 0.001
p_com = np.array([0.1, 0.3, 0.5, 0.7, 0.9]).reshape(-1, 1) * 0.08 + 0.01
p_loc = np.array([0.1, 0.3, 0.5, 0.7, 0.9]).reshape(-1, 1) * 0.4 + 0.1
p_id = np.concatenate([p_id]*multiple)
p_com = np.concatenate([p_com]*multiple)
p_loc = np.concatenate([p_loc]*multiple)
# Coste = 0.007663048839892153
# Costu = 0.037580389962800664
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

C, D, a, Cost_max, Cost_max_index = scheme1(M, N, C_u, Band, pw, N0, h_u, p_id, p_com, p_loc, lamda_e, lamda_t, Din, Dout, omg, f_loc,
                            f_c, prob_change=real_prob, prob_fixed=real_prob)
print(f"C = \n{C}")
print(f"D = \n{D}")
print(f"a = \n{a}")
print(f"Cost_max = \n{Cost_max}")

T, E, Cost_max2 = T_E.algorithm_te(M, N, C_u, Band, pw, N0, h_u, p_id, p_com, p_loc, lamda_e, lamda_t, Din, Dout, omg, f_loc, f_c, prob=real_prob, C=C, D=D, a=a, Cost_max_index=Cost_max_index)
print(f"T_max = {T}")
print(f"E_max = {E}")
print(f"Cost_max = {Cost_max2}")
print(f"Total = {T*0.8+E*0.2-Cost_max}")
# print(f"diff = {Cost_max - Cost_max2}")
