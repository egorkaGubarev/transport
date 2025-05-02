import dimod
import neal
import numpy as np

import config
import create_hamilt
import create_variables
import postproc
import utils

name_to_index = {}
index_to_name = {}

mu = create_variables.create_matrix('mu', (config.stations, config.vehicles), name_to_index, index_to_name)
eta = create_variables.create_matrix('eta', (config.stations, config.vehicles), name_to_index, index_to_name)
x = create_variables.create_tensor( (config.stations, config.stations, config.vehicles),
                                    name_to_index, index_to_name)
slack, subset_to_index, index_to_subset = create_variables.create_slack(config.stations, name_to_index, index_to_name)

target = create_hamilt.create_target(x, mu, eta, config.d_stations, config.d_depots, config.vehicles)
single_out = create_hamilt.create_single_out(config.b, config.stations, x, eta)
single_in = create_hamilt.create_single_in(config.b, config.stations, x, mu)
single_start = create_hamilt.create_single_start(config.b, mu, config.vehicles)
single_end = create_hamilt.create_single_end(config.b, eta, config.vehicles)
continuity = create_hamilt.create_continuity(config.stations, config.b, x, mu, eta, config.vehicles)
sub_tour = create_hamilt.create_sub_tour(subset_to_index, x, slack, config.b)

matrix, _ = (target + single_out + single_in + single_start + single_end + continuity + sub_tour).compile().to_qubo()
best_solution = None
best_target = 100
for solution_dict in neal.SimulatedAnnealingSampler().sample(dimod.BQM(matrix, 'BINARY'),
                                                             num_reads=config.num_reads_d_wave):
    mu = postproc.store_matrix('mu', solution_dict, (config.vehicles, config.stations))
    eta = postproc.store_matrix('eta', solution_dict, (config.vehicles, config.stations))
    x = postproc.store_x(solution_dict, (config.vehicles, config.stations, config.stations))
    slack = postproc.store_lambda(solution_dict, subset_to_index)
    single_out = create_hamilt.create_single_out(config.b, config.stations, x, eta)
    single_in = create_hamilt.create_single_in(config.b, config.stations, x, mu)
    single_start = create_hamilt.create_single_start(config.b, mu, config.vehicles)
    single_end = create_hamilt.create_single_end(config.b, eta, config.vehicles)
    continuity = create_hamilt.create_continuity(config.stations, config.b, x, mu, eta, config.vehicles)
    sub_tour = create_hamilt.create_sub_tour(subset_to_index, x, slack, config.b)
    if single_out + single_in + single_start + single_end + continuity + sub_tour == 0:
        target = create_hamilt.create_target(x, mu, eta, config.d_stations, config.d_depots, config.vehicles)
        if target < best_target:
            best_target = target
            best_solution = solution_dict

print(f'Routes: {utils.find_route(best_solution, config.vehicles, config.stations)}')
print(f'Target: {np.round(best_target, 2)}')
