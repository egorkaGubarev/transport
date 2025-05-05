import numpy as np

import config
import create_hamilt
import create_variables
import utils

name_to_index = {}
index_to_name = {}

mu = create_variables.create_matrix('mu', (config.stations, config.vehicles), name_to_index, index_to_name)
eta = create_variables.create_matrix('eta', (config.stations, config.vehicles), name_to_index, index_to_name)
x = create_variables.create_tensor( (config.stations, config.stations, config.vehicles),
                                    name_to_index, index_to_name)
slack, subset_to_index, index_to_subset = create_variables.create_slack(config.stations, name_to_index, index_to_name)
slack_start = create_variables.create_vector('slack_start', config.vehicles, name_to_index, index_to_name)
slack_end = create_variables.create_vector('slack_end', config.vehicles, name_to_index, index_to_name)

target = create_hamilt.create_target(x, mu, eta, config.d_stations, config.d_depots, config.vehicles)
single_out = create_hamilt.create_single_out(config.b, config.stations, x, eta)
single_in = create_hamilt.create_single_in(config.b, config.stations, x, mu)
single_start = create_hamilt.create_single_start(config.b, mu, slack_start, config.vehicles)
single_end = create_hamilt.create_single_end(config.b, eta, slack_end, config.vehicles)
continuity = create_hamilt.create_continuity(config.stations, config.b, x, mu, eta, config.vehicles)
sub_tour = create_hamilt.create_sub_tour(subset_to_index, x, slack, config.b)

matrix, _ = (target + single_out + single_in + single_start + single_end + continuity + sub_tour).compile().to_qubo()
solution, target = utils.optimize_with_d_wave(matrix, config.num_reads_d_wave, config.vehicles, config.stations,
                                              config.b, config.d_depots, config.d_stations, subset_to_index)

print(f'Routes: {utils.find_route(solution, config.vehicles, config.stations)}')
print(f'Target: {np.round(target, 2)}')
