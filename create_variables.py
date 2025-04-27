import itertools
import numpy as np
import pyqubo

def create_matrix(size, name_to_index, index_to_name):
    matrix = []
    for i in range(size):
        matrix.append(create_vector('x_' + str(i), size, name_to_index, index_to_name, i))
    return np.array(matrix)

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
        slack.append(create_vector('lambda_' + str(subset_index), int(np.ceil(1 + np.log2(cardin - 1))) + 1,
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
