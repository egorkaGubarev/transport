import numpy as np

import utils

def count_sum_lambda_over_s(cardin, slack, si):
    sum_l = 0
    for l in range(utils.count_slack_amount(cardin)):
        sum_l += 2 ** l * slack[si][l]
    return sum_l

def count_sum_x_over_s(x, s):
    sum_x = 0
    for i in s:
        sum_x += np.sum(x[:, i, s])
    return sum_x

def create_continuity(stations, b, x, mu, eta, vehicles):
    constrict = 0
    for k in range(vehicles):
        constrict += create_continuity_for_vehicle(stations, b, x, mu, eta, k)
    return constrict

def create_continuity_for_vehicle(stations, b, x, mu, eta, k):
    constrict = 0
    for i in range(stations):
        constrict += b * (np.sum(x[k, :, i]) + mu[k, i] - np.sum(x[k, i, :]) - eta[k, i]) ** 2
    return constrict

def create_single_end(b, eta, vehicles):
    constrict = 0
    for k in range(vehicles):
        constrict += b * (1 - np.sum(eta[k])) ** 2
    return constrict

def create_single_in(b, stations, x, mu):
    constrict = 0
    for i in range(stations):
        constrict += b * (1 - (np.sum(x[:, :, i]) + np.sum(mu[:, i]))) ** 2
    return constrict

def create_single_out(b, stations, x, eta):
    constrict = 0
    for i in range(stations):
        constrict += b * (1 - (np.sum(x[:, i, :]) + np.sum(eta[:, i]))) ** 2
    return constrict

def create_single_start(b, mu, vehicles):
    constrict = 0
    for k in range(vehicles):
        constrict += b * (1 - np.sum(mu[k])) ** 2
    return constrict

def create_sub_tour(subset_to_index, x, slack, b, debug=False):
    constrict = 0
    for s, si in subset_to_index.items():
        cardin = len(s)
        delta = b * (count_sum_x_over_s(x, s) + count_sum_lambda_over_s(cardin, slack, si) - cardin + 1) ** 2
        if debug and delta > 0:
            print(f'Subset: {s}')
            print(f'Subset index: {si}')
            print(f'Delta: {delta}')
        constrict += delta
    return constrict

def create_target(x, mu, eta, d_stations, d_depots, vehicles):
    target = 0
    for k in range(vehicles):
        target += create_target_for_vehicle(x, mu, eta,  d_stations, d_depots, k)
    return target

def create_target_for_vehicle(x, mu, eta, d_stations, d_depots, k):
    return np.sum(d_stations * x[k]) + np.sum(d_depots * (mu[k] + eta[k]))
