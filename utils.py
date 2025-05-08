import dimod
import neal
import numpy as np

import create_hamilt
import postproc


def count_visual_matrix(matrix, name_to_index):
    variables = len(name_to_index)
    visual_matrix = np.zeros((variables, variables))
    for pair, value in matrix.items():
        visual_matrix[name_to_index[pair[0]], name_to_index[pair[1]]] = value
    return visual_matrix


def find_next_for_vehicle(previous, config, k):
    found = False
    station = 0
    prefix = 'x_' + str(k) + '_' + str(previous) + '_'
    while not found:
        if station == previous:
            station += 1
        if config[prefix + str(station)] == 1:
            found = True
        else:
            station += 1
    return station


def find_route(config, vehicles, stations):
    route = []
    for k in range(vehicles):
        route.append(find_route_for_vehicle(config, k, stations))
    return route


def find_route_for_vehicle(config, k, stations):
    station_found, station = find_station_for_vehicle('mu', config, k, stations)
    if station_found:
        route = [station]
        _, last = find_station_for_vehicle('eta', config, k, stations)
        while station != last:
            route.append(station := find_next_for_vehicle(station, config, k))
    else:
        route = []
    return route


def find_station_for_vehicle(prefix, config, k, stations):
    station_found = False
    station = 0
    prefix_for_vehicle = prefix + '_' + str(k) + '_'
    while not station_found and station < stations:
        if config[prefix_for_vehicle + str(station)] == 1:
            station_found = True
        else:
            station += 1
    return station_found, station


def save_config(config, name_to_index):
    config_dict = {}
    for name, index in name_to_index.items():
        config_dict[name] = config[index]
    return config_dict


def count_distance(routes, d_stations, d_depots):
    distance = 0
    for route in routes:
        distance += count_distance_for_vehicle(route, d_stations, d_depots)
    return distance


def count_distance_for_vehicle(route, d_stations, d_depots):
    previous = route[0]
    distance = d_depots[previous]
    for i in range(1, len(route)):
        station = route[i]
        distance += d_stations[previous, station]
        previous = station
    distance += d_depots[route[-1]]
    return distance


def count_slack_amount(limit):
    return int(np.ceil(np.log2(limit + 1)))


def force_solution(solution, my_solution):
    for variable in solution:
        if variable in my_solution:
            solution[variable] = my_solution[variable]
        else:
            solution[variable] = 0
    return solution


def optimize_with_d_wave(matrix, num_reads, vehicles, stations, b,
                         d_depots, d_stations, capac, demand, gamma, depot_capac, depots, subset_to_index):
    best_solution = None
    best_target = 100
    for solution_dict in neal.SimulatedAnnealingSampler().sample(dimod.BQM(matrix, 'BINARY'),
                                                                 num_reads=num_reads):
        mu = postproc.store_matrix('mu', solution_dict, (vehicles, stations))
        eta = postproc.store_matrix('eta', solution_dict, (vehicles, stations))
        x = postproc.store_x(solution_dict, (vehicles, stations, stations))
        slack = postproc.store_lambda(solution_dict, subset_to_index)
        slack_start = postproc.store_vector('slack_start', vehicles, solution_dict)
        slack_end = postproc.store_vector('slack_end', vehicles, solution_dict)
        slack_capac = postproc.store_matrix('slack_capac', solution_dict,
                                            (vehicles, count_slack_amount(np.max(capac))))
        slack_depot_capac = postproc.store_matrix('slack_depot_capac', solution_dict,
                                                  (depots, count_slack_amount(np.max(depot_capac))))
        single_out = create_hamilt.create_single_out(b, stations, x, eta)
        single_in = create_hamilt.create_single_in(b, stations, x, mu)
        single_start = create_hamilt.create_single_start(b, mu, slack_start, vehicles)
        single_end = create_hamilt.create_single_end(b, eta, slack_end, vehicles)
        continuity = create_hamilt.create_continuity(stations, b, x, mu, eta, vehicles)
        sub_tour = create_hamilt.create_sub_tour(subset_to_index, x, slack, b)
        demand_constrict = create_hamilt.create_demand(x, eta, demand, stations, slack_capac, capac, vehicles, b)
        depot_capac_constrict = create_hamilt.create_depot_capac(x, demand, eta, gamma, slack_depot_capac,
                                                                 stations, depots, vehicles, depot_capac, b)
        if (single_out + single_in + single_start + single_end +
                continuity + sub_tour + demand_constrict + depot_capac_constrict == 0):
            target = create_hamilt.create_target(x, mu, eta, d_stations, d_depots, gamma, vehicles)
            if target < best_target:
                best_target = target
                best_solution = solution_dict
    return best_solution, best_target


def print_dict(data):
    for variable, value in data.items():
        print(f'{variable}: {value}')
