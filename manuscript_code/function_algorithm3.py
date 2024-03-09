import numpy as np
import math
import function_algorithm1
import function_algorithm2


def algorithm3(M, N, C_u, Band, pw, N0, h_u, p_id, p_com, p_loc, lamda_e, lamda_t, Din, Dout, omg, f_loc, f_c, prob_change, prob_fixed):
    a = np.ones([M]).reshape(-1, 1) / M
    # Cost_max = math.pow(10, 10)
    Cost_max = 0.1
    Cost_opt = -0.1
    iteration1 = 0

    while np.fabs(Cost_max - Cost_opt) != math.pow(10, -12):
        iteration1 += 1
        temp = Cost_max - Cost_opt
        print(temp)
        Cost_opt = Cost_max
        C, D = function_algorithm1.algorithm1(M, N, C_u, Band, a, pw, N0, h_u, p_id, p_com, p_loc, lamda_e, lamda_t,
                                              Din, Dout, omg, f_loc, f_c, prob_change)
        a, Cost_max, Cost_max_index = function_algorithm2.algorithm2(M, N, C_u, Band, pw, N0, h_u, p_id, p_com, p_loc, lamda_e, lamda_t,
                                                     Din, Dout, omg, f_loc, f_c, prob_fixed, C, D)
        aaa=Cost_max - Cost_opt
        if np.fabs(Cost_max - Cost_opt) <= math.pow(10, -4):
            print(Cost_max - Cost_opt)
        # if Cost_max == Cost_opt:

        #
            return C, D, a, Cost_max, Cost_max_index


# M = 5
# N = 10
# C_u = 4
# Band = 30 * math.pow(10, 6)
# pw = 0.1
# N0 = -174
# h_u = np.ones(M).reshape(-1, 1) * 2
# p_id = np.array([0.1, 0.3, 0.5, 0.7, 0.9]).reshape(-1, 1) * 0.008 + 0.001
# p_com = np.array([0.1, 0.3, 0.5, 0.7, 0.9]).reshape(-1, 1) * 0.08 + 0.01
# p_loc = np.array([0.1, 0.3, 0.5, 0.7, 0.9]).reshape(-1, 1) * 0.4 + 0.1
# lamda_e = 1
# lamda_t = 1
# Din = 3 * math.pow(10, 6)
# Dout = (np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]).reshape(1, -1) * 2 + 6) * math.pow(10, 6)
# omg = np.array([0.05, 0.15, 0.95, 0.65, 0.45, 0.55, 0.35, 0.75, 0.85, 0.25]).reshape(1, -1) * 10 + 10
# f_loc = np.array([0.75, 0.5, 1.5, 1.25, 1]).reshape(-1, 1) * np.power(10, 9)
# f_c = 4 * math.pow(10, 9)
#
# prob = np.array([
#     [0.06390299, 0.0873196, 0.18882476, 0.17413469, 0.14298607, 0.0124893, 0.05516967, 0.16545752, 0.09141752,
#      0.01829789],
#     [0.06848021, 0.02462554, 0.0925854, 0.10605827, 0.17757285, 0.14807804, 0.17844337, 0.01150095, 0.02469164,
#      0.16796372],
#     [0.07688904, 0.15797693, 0.15064112, 0.07820362, 0.14451828, 0.03227132, 0.11202526, 0.16865301, 0.02764718,
#      0.05117423],
#     [0.16464213, 0.0478535, 0.04451598, 0.14514572, 0.12180596, 0.03692565, 0.09997204, 0.17170378, 0.07541009,
#      0.09202515],
#     [0.11246995, 0.02348793, 0.10510097, 0.09418531, 0.05586772, 0.09828553, 0.14064972, 0.0845822, 0.14966726,
#      0.1357034]
# ])
#
# C, D, a, Cost_max = algorithm3(M, N, C_u, Band, pw, N0, h_u, p_id, p_com, p_loc, lamda_e, lamda_t, Din, Dout, omg, f_loc, f_c, prob_change=prob, prob_fixed=prob)
# print(f"C = \n{C}")
# print(f"D = \n{D}")
# print(f"a = \n{a}")
# print(f"Cost_max = \n{Cost_max}")

