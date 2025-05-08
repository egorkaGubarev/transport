import numpy as np

stations = 4
vehicles = 3
depots = 2
d_stations = np.array([[0, 1, np.sqrt(2), 2],
                       [1, 0, 1, 3],
                       [np.sqrt(2), 1, 0, np.sqrt(10)],
                       [2, 3, np.sqrt(10), 0]])
gamma = np.array([[1, 0],
                  [0, 1],
                  [1, 0]])
d_depots = np.array([[1, 2, np.sqrt(5), 1],
                     [2, 1, np.sqrt(2), 4]])
demand = np.array([4, 2, 1, 3])
capac = np.array([4, 6, 6])
depot_capac = np.array([4, 6])

m_budget = 50000
num_reads = 10000
num_reads_d_wave = 1000
b = 10
