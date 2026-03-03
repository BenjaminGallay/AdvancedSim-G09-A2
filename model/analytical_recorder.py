mean_travel_delay = 0
road_length = 0
vehicle_speed = 50 * 1000 / 60


def reset():
    global mean_travel_delay
    mean_travel_delay = 0
    global road_length
    road_length = 0
    return


def bridge_delay_record(expected_mean_delay):
    global mean_travel_delay
    mean_travel_delay += expected_mean_delay
    return


def road_length_record(length):
    global road_length
    road_length += length


def get_expected_mean_travel_time():
    return mean_travel_delay + road_length / vehicle_speed
