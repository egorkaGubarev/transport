def count_target(x, mu, eta, d_stations, d_depots, vehicles):
    target = 0
    for k in range(vehicles):
        target += create_target_for_vehicle(x, mu, eta,  d_stations, d_depots, k)
    return target