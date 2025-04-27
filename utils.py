import numpy as np

def count_visual_matrix(matrix, name_to_index):
    variables = len(name_to_index)
    visual_matrix = np.zeros((variables, variables))
    for pair, value in matrix.items():
        visual_matrix[name_to_index[pair[0]], name_to_index[pair[1]]] = value
    return visual_matrix

def find_next(previous, config, name_to_index):
  found = False
  station = 0
  prefix = 'x_' + str(previous) + '_'
  while not found:
    if station == previous:
      station += 1
    if config[name_to_index[prefix + str(station)]] == 1:
        found = True
    else:
      station += 1
  return station

def find_route(config, name_to_index):
    route = [station := find_station('mu', config, name_to_index)]
    last = find_station('eta', config, name_to_index)
    while station != last:
        route.append(station := find_next(station, config, name_to_index))
    return route

def find_station(prefix, config, name_to_index):
  station_found = False
  station = 0
  while not station_found:
    if config[name_to_index[prefix + '_' + str(station)]] == 1:
      station_found = True
    else:
      station += 1
  return station
