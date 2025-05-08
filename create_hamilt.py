import numpy as np

import utils


def count_sum_lambda_over_s(cardin, slack, si):
    sum_l = 0
    for digit in range(utils.count_slack_amount(cardin)):
        sum_l += 2 ** digit * slack[si][digit]
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


def create_demand(x, eta, demand, stations, slack, capac, vehicles, b):
    constrict = 0
    for k in range(vehicles):
        constrict += b * (create_demand_for_vehicle(x, eta, demand, stations, k, slack[k], capac[k])) ** 2
    return constrict


def create_demand_for_vehicle(x, eta, demand, stations, k, slack, capac):
    constrict = 0
    for j in range(stations):
        constrict += np.sum(demand * x[k, :, j])
    constrict += np.sum(demand * eta[k])
    for digit in range(len(slack)):
        constrict += 2 ** digit * slack[digit]
    return constrict - capac


def create_depot_capac(x, demand, eta, gamma, slack, stations, depots, vehicles, depot_capac, b):
    constrict = 0
    for d in range(depots):
        constrict += b * create_depot_capac_for_depot(x, demand, eta, gamma, slack[d],
                                                      stations, vehicles, depot_capac, d) ** 2
    return constrict


def create_depot_capac_for_depot(x, demand, eta, gamma, slack, stations, vehicles, depot_capac, d):
    constrict = 0
    for k in range(vehicles):
        constrict += create_depot_capac_for_vehicle(x, demand, eta, gamma, stations, k, d)
    for digit in range(len(slack)):
        constrict += 2 ** digit * slack[digit]
    return constrict - depot_capac[d]


def create_depot_capac_for_vehicle(x, demand, eta, gamma, stations, k, d):
    constrict = 0
    for j in range(stations):
        constrict += gamma[k, d] * np.sum(demand * x[k, :, j])
    constrict += gamma[k, d] * np.sum(demand * eta[k])
    return constrict


def create_single_end(b, eta, slack, vehicles):
    constrict = 0
    for k in range(vehicles):
        constrict += b * (1 - np.sum(eta[k]) - slack[k]) ** 2
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


def create_single_start(b, mu, slack, vehicles):
    constrict = 0
    for k in range(vehicles):
        constrict += b * (1 - np.sum(mu[k]) - slack[k]) ** 2
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


def create_target(x, mu, eta, d_stations, d_depots, gamma, vehicles):
    target = 0
    for k in range(vehicles):
        target += create_target_for_vehicle(x, mu, eta,  d_stations, d_depots, gamma, k)
    return target


def create_target_for_vehicle(x, mu, eta, d_stations, d_depots, gamma, k):
    return np.sum(d_stations * x[k]) + np.sum(np.matmul(gamma[k], d_depots) * (mu[k] + eta[k]))
