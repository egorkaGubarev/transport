import qdeepsdk

import config
import create_hamilt
import create_variables
import q_deep_token
import utils

name_to_index = {}
index_to_name = {}

mu = create_variables.create_vector('mu', config.stations, name_to_index, index_to_name)
eta = create_variables.create_vector('eta', config.stations, name_to_index, index_to_name)
x = create_variables.create_matrix(config.stations, name_to_index, index_to_name)
slack, subset_to_index, index_to_subset = create_variables.create_slack(config.stations, name_to_index, index_to_name)

target = create_hamilt.create_target(x, mu, eta, config.d_stations, config.d_depots)
single_out = create_hamilt.create_single_out(config.b, config.stations, x, eta)
single_in = create_hamilt.create_single_in(config.b, config.stations, x, mu)
single_start = create_hamilt.create_single_start(config.b, mu)
single_end = create_hamilt.create_single_end(config.b, eta)
continuity = create_hamilt.create_continuity(config.stations, config.b, x)
sub_tour = create_hamilt.create_sub_tour(subset_to_index, x, slack, config.b_sub_tour)

solver = qdeepsdk.QDeepHybridSolver()
solver.token = q_deep_token.token
solver.m_budget = config.m_budget
solver.num_reads = config.num_reads

matrix, _ = (target + single_out + single_in + single_start + single_end + continuity + sub_tour).compile().to_qubo()
solution = solver.solve(utils.count_visual_matrix(matrix, name_to_index))['QdeepHybridSolver']['configuration']
print(utils.find_route(solution, name_to_index))
