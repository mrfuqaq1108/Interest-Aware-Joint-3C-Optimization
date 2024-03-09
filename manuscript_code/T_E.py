import numpy as np
import math


def algorithm_te(M, N, C_u, Band, pw, N0, h_u, p_id, p_com, p_loc, lamda_e, lamda_t, Din, Dout, omg, f_loc, f_c, prob, C, D, a, Cost_max_index):
    r = a * Band * np.log2(1 + pw * h_u * math.pow(10, -6) / (math.pow(10, N0 / 10) / 1000))
    At = prob * Din / r
    Bt = prob * Din * omg / f_loc
    Ft = prob * Din * omg / f_c
    Gt = prob * Dout / r
    Ht = Ft + Gt

    T_temp = -At * C + (At + Bt - Ft - Gt) * D + Ht
    T_u = np.sum(T_temp, axis=1).reshape(-1, 1)
    T = T_u[Cost_max_index]

    Ae = prob * p_com * Din / r
    Be = prob * p_loc * Din * omg / f_loc
    Fe = prob * p_id * Din * omg / f_c
    Ge = prob * p_com * Dout / r
    He = Fe + Ge

    E_temp = -Ae * C + (Ae + Be - Fe - Ge) * D + He
    E_u = np.sum(E_temp, axis=1).reshape(-1, 1)
    E = E_u[Cost_max_index]

    A = prob * (lamda_e * p_com + lamda_t) * Din / r
    B = prob * (lamda_e * p_loc + lamda_t) * Din * omg / f_loc
    F = prob * (lamda_e * p_id + lamda_t) * Din * omg / f_c
    G = prob * (lamda_e * p_com + lamda_t) * Dout / r
    H = F + G

    Cost_temp = -A * C + (A + B - F - G) * D + H
    Cost_u = np.sum(Cost_temp, axis=1).reshape(-1, 1)
    Cost_max = np.max(Cost_u)




    return T, E, Cost_max
