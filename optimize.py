import numpy as np
import qdeepsdk

import config
import create_hamilt
import create_variables
import postproc
import q_deep_token
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

solver = qdeepsdk.QDeepHybridSolver()
solver.token = q_deep_token.token
solver.m_budget = config.m_budget
solver.num_reads = config.num_reads

matrix, _ = (target + single_out + single_in + single_start + single_end + continuity + sub_tour).compile().to_qubo()
response = solver.solve(utils.count_visual_matrix(matrix, name_to_index))['QdeepHybridSolver']
print(response)
solution_dict = utils.save_config(response['configuration'], name_to_index)
correct = {'mu_0_0': 1, 'x_0_0_1': 1, 'x_0_1_2': 1, 'x_0_2_3': 1, 'eta_0_3': 1,
           'lambda_1_0': 1, 'lambda_2_0': 1, 'lambda_4_0': 1, 'lambda_7_0': 1, 'lambda_8_0': 1}
# utils.force_solution(solution_dict, correct)
routes = utils.find_route(solution_dict, config.vehicles, config.stations)
mu = postproc.store_matrix('mu', solution_dict, (config.vehicles, config.stations))
eta = postproc.store_matrix('eta', solution_dict, (config.vehicles, config.stations))
x = postproc.store_x(solution_dict, (config.vehicles, config.stations, config.stations))
slack = postproc.store_lambda(solution_dict, subset_to_index)
utils.print_dict(solution_dict)
utils.print_dict(subset_to_index)
print(f'Target: {np.round(create_hamilt.create_target(x, mu, eta,
                                                      config.d_stations, config.d_depots,
                                                      config.vehicles),2)}')
print(f'Single out: {np.round(create_hamilt.create_single_out(config.b, config.stations, x, eta), 2)}')
print(f'Single in: {np.round(create_hamilt.create_single_in(config.b, config.stations, x, mu), 2)}')
print(f'Single start: {np.round(create_hamilt.create_single_start(config.b, mu, config.vehicles), 2)}')
print(f'Single end: {np.round(create_hamilt.create_single_end(config.b, eta, config.vehicles), 2)}')
print(f'Continuity: {np.round(create_hamilt.create_continuity(config.stations, config.b, x,
                                                              mu, eta, config.vehicles), 2)}')
print(f'Sub_tour: {np.round(create_hamilt.create_sub_tour(subset_to_index, x, slack,
                                                          config.b, True), 2)}')
print(f'Route: {routes}')
print(f'Distance: {np.round(utils.count_distance(routes, config.d_stations, config.d_depots), 2)}')
print(f'Energy: {np.round(response['energy'], 2)}')
