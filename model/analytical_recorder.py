mean_travel_delay = 0
road_length = 0
vehicle_speed = 50 * 1000 / 60


def reset():
    global mean_travel_delay
    mean_travel_delay = 0
    global road_length
    road_length = 0
    return


def bridge_delay_record(length, condition, breakdown_probabilities):
    if length < 10:
        mean_delay = 15
    elif length < 50:
        mean_delay = 37.5
    elif length < 200:
        mean_delay = 67.5
    else:
        mean_delay = (
            7 / 3
        ) * 60  # expected value of the triangular distribution of probability
    delay_probability = breakdown_probabilities[condition]

    global mean_travel_delay
    mean_travel_delay += mean_delay * delay_probability
    return


def road_length_record(length):
    global road_length
    road_length += length


def get_expected_mean_travel_time():
    return mean_travel_delay + road_length / vehicle_speed
