import itertools
import numpy as np
import pyqubo

import utils

def create_matrix(name, size, name_to_index, index_to_name):
    matrix = []
    prefix = name + '_'
    for i in range(size[1]):
        matrix.append(create_vector(prefix + str(i), size[0], name_to_index, index_to_name))
    return np.array(matrix)

def create_nondiagonal_matrix(name, size, name_to_index, index_to_name):
    matrix = []
    prefix = name + '_'
    for i in range(size[0]):
        matrix.append(create_vector(prefix + str(i), size[1], name_to_index, index_to_name, i))
    return np.array(matrix)

def create_tensor(size, name_to_index, index_to_name):
    tensor = []
    for k in range(size[2]):
        tensor.append(create_nondiagonal_matrix('x_' + str(k), (size[0], size[1]), name_to_index, index_to_name))
    return np.array(tensor)

def create_slack(stations_amount, name_to_index, index_to_name):
    stations = np.arange(stations_amount)
    slack = []
    subset_to_index = {}
    index_to_subset = {}
    for cardin in range(2, stations_amount + 1):
        create_slack_for_cardin(stations, cardin, slack, subset_to_index, index_to_subset, name_to_index, index_to_name)
    return slack, subset_to_index, index_to_subset

def create_slack_for_cardin(stations, cardin, slack, subset_to_index, index_to_subset, name_to_index, index_to_name):
    for subset in itertools.combinations(stations, cardin):
        subset_index = len(index_to_subset)
        store_name_index(subset, subset_index, subset_to_index, index_to_subset)
        slack.append(create_vector('lambda_' + str(subset_index), utils.count_slack_amount(cardin),
                                   name_to_index, index_to_name))

def create_vector(name, length, name_to_index, index_to_name, skip=None):
    vector = []
    prefix = name + '_'
    for i in range(length):
        if i != skip:
            store_variable(prefix + str(i), name_to_index, index_to_name, vector)
        else:
            vector.append(0)
    return np.array(vector)

def store_name_index(name, index, name_to_index, index_to_name):
    name_to_index[name] = index
    index_to_name[index] = name

def store_variable(name, name_to_index, index_to_name, store):
    index = len(name_to_index)
    store_name_index(name, index, name_to_index, index_to_name)
    store.append(pyqubo.Binary(name))
