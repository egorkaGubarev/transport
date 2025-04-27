import numpy as np

def count_sum_lambda_over_s(cardin, slack, si):
    sum_l = 0
    for l in range(int(np.ceil(1 + np.log2(cardin - 1))) + 1):
        sum_l += 2 ** l * slack[si][l]
    return sum_l

def count_sum_x_over_s(x, s):
    sum_x = 0
    for i in s:
        sum_x += np.sum(x[i, s])
    return sum_x

def create_continuity(stations, b, x):
    constrict = 0
    for i in range(stations):
        constrict += b * (np.sum(x[:, i]) - np.sum(x[i, :])) ** 2
    return constrict

def create_single_end(b, eta):
    return b * (1 - np.sum(eta)) ** 2

def create_single_in(b, stations, x, mu):
    constrict = 0
    for i in range(stations):
        constrict += b * (1 - (np.sum(x[:, i]) + mu[i])) ** 2
    return constrict

def create_single_out(b, stations, x, eta):
    constrict = 0
    for i in range(stations):
        constrict += b * (1 - (np.sum(x[i, :]) + eta[i])) ** 2
    return constrict

def create_single_start(b, mu):
    return b * (1 - np.sum(mu)) ** 2

def create_sub_tour(subset_to_index, x, slack, b):
    constrict = 0
    for s, si in subset_to_index.items():
        cardin = len(s)
        constrict += b * (count_sum_x_over_s(x, s) + count_sum_lambda_over_s(cardin, slack, si) - cardin + 1) ** 2
    return constrict

def create_target(x, mu, eta, d_stations, d_depots):
    return np.sum(d_stations * x) + np.sum(d_depots * (mu + eta))
