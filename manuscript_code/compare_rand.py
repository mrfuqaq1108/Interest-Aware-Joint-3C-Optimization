import numpy as np
import math
import function_algorithm3
import para

M = 5
N = 10
C_u = 4
Band = 30 * math.pow(10, 6)
pw = 0.1
N0 = -174
h_u = np.ones(M).reshape(-1, 1) * 2
p_id = np.array([0.1, 0.3, 0.5, 0.7, 0.9]).reshape(-1, 1) * 0.008 + 0.001
p_com = np.array([0.1, 0.3, 0.5, 0.7, 0.9]).reshape(-1, 1) * 0.08 + 0.01
p_loc = np.array([0.1, 0.3, 0.5, 0.7, 0.9]).reshape(-1, 1) * 0.4 + 0.1
Coste = 1
Costu = 1
lamda_e = 0.2 / Coste
lamda_t = 0.8 / Costu
Din = 3 * math.pow(10, 6)
Dout = (np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]).reshape(1, -1) * 2 + 6) * math.pow(10, 6)
omg = np.array([0.05, 0.15, 0.95, 0.65, 0.45, 0.55, 0.35, 0.75, 0.85, 0.25]).reshape(1, -1) * 10 + 10
f_loc = np.array([0.75, 0.5, 1.5, 1.25, 1]).reshape(-1, 1) * np.power(10, 9)
f_c = 2 * math.pow(10, 9)

rand_prob = para.random_prob
real_prob = para.real_prob
C, D, a, Cost_max = function_algorithm3.algorithm3(M, N, C_u, Band, pw, N0, h_u, p_id, p_com, p_loc, lamda_e, lamda_t, Din, Dout, omg, f_loc, f_c, prob_change=rand_prob, prob_fixed=real_prob)
print(f"C = \n{C}")
print(f"D = \n{D}")
print(f"a = \n{a}")
print(f"Cost_max = \n{Cost_max}")

# 0.03561701396634619
