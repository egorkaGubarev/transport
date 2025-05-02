import numpy as np

def count_visual_matrix(matrix, name_to_index):
    variables = len(name_to_index)
    visual_matrix = np.zeros((variables, variables))
    for pair, value in matrix.items():
        visual_matrix[name_to_index[pair[0]], name_to_index[pair[1]]] = value
    return visual_matrix

def find_next_for_vehicle(previous, config, k):
  found = False
  station = 0
  prefix = 'x_' + str(k) + '_' + str(previous) + '_'
  while not found:
    if station == previous:
      station += 1
    if config[prefix + str(station)] == 1:
        found = True
    else:
      station += 1
  return station

def find_route(config, vehicles, stations):
    route = []
    for k in range(vehicles):
        route.append(find_route_for_vehicle(config, k, stations))
    return route

def find_route_for_vehicle(config, k, stations):
    station_found, station = find_station_for_vehicle('mu', config, k, stations)
    if station_found:
        route = [station]
        _, last = find_station_for_vehicle('eta', config, k, stations)
        while station != last:
            route.append(station := find_next_for_vehicle(station, config, k))
    else:
        route = []
    return route

def find_station_for_vehicle(prefix, config, k, stations):
  station_found = False
  station = 0
  prefix_for_vehicle = prefix + '_' + str(k) + '_'
  while not station_found and station < stations:
    if config[prefix_for_vehicle + str(station)] == 1:
      station_found = True
    else:
      station += 1
  return station_found, station

def save_config(config, name_to_index):
    config_dict = {}
    for name, index in name_to_index.items():
        config_dict[name] = config[index]
    return config_dict

def count_distance(routes, d_stations, d_depots):
    distance = 0
    for route in routes:
        distance += count_distance_for_vehicle(route, d_stations, d_depots)
    return distance

def count_distance_for_vehicle(route, d_stations, d_depots):
    previous = route[0]
    distance = d_depots[previous]
    for i in range(1, len(route)):
        station = route[i]
        distance += d_stations[previous, station]
        previous = station
    distance += d_depots[route[-1]]
    return distance

def count_slack_amount(cardin):
    return int(np.ceil(np.log2(cardin + 1)))

def force_solution(solution, my_solution):
    for variable in solution:
        if variable in my_solution:
            solution[variable] = my_solution[variable]
        else:
            solution[variable] = 0
    return solution

def print_dict(data):
    for variable, value in data.items():
        print(f'{variable}: {value}')
