import numpy as np

import utils


def store_matrix(name_to_store, solution, size):
    my_array = np.zeros(size)
    name_length = len(name_to_store)
    for name, value in solution.items():
        if name[:name_length] == name_to_store:
            index = list(map(int, name[name_length:].split('_')[1:]))
            my_array[index[0]][index[1]] = value
    return my_array


def store_x(solution, size):
    x = np.zeros(size)
    for k in range(size[0]):
        x[k] = store_matrix('x_' + str(k), solution, (size[1], size[2]))
    return x


def store_lambda(solution, subset_to_index):
    slack = []
    for subset, index in subset_to_index.items():
        slack.append(store_vector('lambda_' + str(index), utils.count_slack_amount(len(subset)), solution))
    return slack


def store_vector(name, length, solution):
    slack = []
    prefix = name + '_'
    for l in range(length):
        slack.append(solution[prefix + str(l)])
    return slack
