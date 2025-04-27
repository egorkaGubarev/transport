import numpy as np

stations = 3
d_stations = np.array([[0, 1, np.sqrt(2)],
                       [1, 0, 1],
                       [np.sqrt(2), 1, 0]])
d_depots = np.array([1, 2, np.sqrt(5)])

m_budget = 1
num_reads = 1
b = 10
b_sub_tour = 100
