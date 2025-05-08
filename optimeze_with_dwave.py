import numpy as np

import config
import create_hamilt
import create_variables
import utils

name_to_index = {}
index_to_name = {}

mu = create_variables.create_matrix('mu', (config.stations, config.vehicles), name_to_index, index_to_name)
eta = create_variables.create_matrix('eta', (config.stations, config.vehicles), name_to_index, index_to_name)
x = create_variables.create_tensor((config.stations, config.stations, config.vehicles),
                                   name_to_index, index_to_name)
slack, subset_to_index, index_to_subset = create_variables.create_slack(config.stations, name_to_index, index_to_name)
slack_start = create_variables.create_vector('slack_start', config.vehicles, name_to_index, index_to_name)
slack_end = create_variables.create_vector('slack_end', config.vehicles, name_to_index, index_to_name)
slack_capac = create_variables.create_slack_matrix('slack_capac', config.vehicles, config.capac,
                                                   name_to_index, index_to_name)
slack_depot_capac = create_variables.create_slack_matrix('slack_depot_capac', config.depots, config.depot_capac,
                                                         name_to_index, index_to_name)

target = create_hamilt.create_target(x, mu, eta, config.d_stations, config.d_depots, config.gamma, config.vehicles)
single_out = create_hamilt.create_single_out(config.b, config.stations, x, eta)
single_in = create_hamilt.create_single_in(config.b, config.stations, x, mu)
single_start = create_hamilt.create_single_start(config.b, mu, slack_start, config.vehicles)
single_end = create_hamilt.create_single_end(config.b, eta, slack_end, config.vehicles)
continuity = create_hamilt.create_continuity(config.stations, config.b, x, mu, eta, config.vehicles)
sub_tour = create_hamilt.create_sub_tour(subset_to_index, x, slack, config.b)
demand = create_hamilt.create_demand(x, eta, config.demand, config.stations,
                                     slack_capac, config.capac, config.vehicles, config.b)
depot_capac = create_hamilt.create_depot_capac(x, config.demand, eta, config.gamma, slack_depot_capac,
                                               config.stations, config.depots, config.vehicles,
                                               config.depot_capac, config.b)

matrix, _ = (target + single_out + single_in + single_start +
             single_end + continuity + sub_tour + demand + depot_capac).compile().to_qubo()
solution, target = utils.optimize_with_d_wave(matrix, config.num_reads_d_wave, config.vehicles, config.stations,
                                              config.b, config.d_depots, config.d_stations,
                                              config.capac, config.demand, config.gamma,
                                              config.depot_capac, config.depots, subset_to_index)

print(f'Routes: {utils.find_route(solution, config.vehicles, config.stations)}')
print(f'Target: {np.round(target, 2)}')
