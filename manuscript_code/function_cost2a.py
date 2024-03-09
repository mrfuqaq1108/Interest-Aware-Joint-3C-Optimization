import numpy as np
import math


def Cost2a(M, Band, pw, N0, h_u, p_id, p_com, p_loc, lamda_e, lamda_t, Din, Dout, omg, f_loc, f_c, prob, C, D, Cost_target):
    amin = math.pow(10, -10) * np.ones(M)
    amax = 1 * np.ones(M)
    a_mat = []
    for u in range(M):
        for iter in range(10000):
            a = (amin[u]+amax[u]) / 2
            ru = a * Band * np.log2(1 + pw * h_u[u][0] * math.pow(10, -6) / (math.pow(10, N0 / 10) / 1000))
            Au = prob[u] * (lamda_e * p_com[u][0] + lamda_t) * Din / ru
            Bu = prob[u] * (lamda_e * p_loc[u][0] + lamda_t) * Din * omg / f_loc[u][0]
            Fu = prob[u] * (lamda_e * p_id[u][0] + lamda_t) * Din * omg / f_c
            Gu = prob[u] * (lamda_e * p_com[u][0] + lamda_t) * Dout / ru
            Hu = Fu + Gu
            Cost_temp2 = -Au*C[u] + (Au+Bu-Fu-Gu)*D[u] + Hu
            Cost_u2 = np.sum(Cost_temp2)
            if np.fabs(Cost_u2-Cost_target) < math.pow(10, -14) or (iter == 9999):
                # print(a)
                # print(iter)
                a_mat.append(a)
                break
            elif Cost_u2 < Cost_target:
                amax[u] = a
            elif Cost_u2 > Cost_target:
                amin[u] = a
    a_mat = np.array(a_mat).reshape(-1, 1)
    return a_mat


# =====/=====/=====/=====/=====/=====/=====/=====/Parameter/=====/=====/=====/=====/=====/=====/=====/=====/=====/=====/

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
# lamda_e = 0
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
# C = np.array([
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 1, 0, 1, 1, 0, 0],
#     [1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
#
# D = np.array([
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 0, 0, 1, 1, 1, 0, 0, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
#
# Cost_target = 0.03737728374432424

# =====/=====/=====/=====/=====/=====/=====/=====/examine/=====/=====/=====/=====/=====/=====/=====/=====/=====/=====/

# a_mat = Cost2a(M, Band, pw, N0, h_u, p_id, p_com, p_loc, lamda_e, lamda_t, Din, Dout, omg, f_loc, f_c, prob, C, D, Cost_target)
# print(a_mat)

# a = np.array([
#     [0.19951117865265555],
#     [0.19583762832546076],
#     [0.11918375404761428],
#     [0.17038757826773404],
#     [0.20000000000000995]
# ])
# for u in range(M):
#     ru = a[u][0] * Band * np.log2(1 + pw * h_u[u][0] * math.pow(10, -6) / (math.pow(10, N0 / 10) / 1000))
#     Au = prob[u] * (lamda_e * p_com[u][0] + lamda_t) * Din / ru
#     Bu = prob[u] * (lamda_e * p_loc[u][0] + lamda_t) * Din * omg / f_loc[u][0]
#     Fu = prob[u] * (lamda_e * p_id[u][0] + lamda_t) * Din * omg / f_c
#     Gu = prob[u] * (lamda_e * p_loc[u][0] + lamda_t) * Dout / ru
#     Hu = Fu + Gu
#     Cost_temp2 = -Au * C[u] + (Au + Bu - Fu - Gu) * D[u] + Hu
#     Cost_u2 = np.sum(Cost_temp2)
#     print(Cost_u2)



