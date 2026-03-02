import os
import statistics

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ids = []
travel_times = []
frequencies = [0 for i in range(2000)]
bridge_waited_time = [0 for i in range(125)]


def reset_times():
    global ids
    ids = []
    global travel_times
    travel_times = []
    return


def truck_record(id, generated_at, removed_at):
    travel_time = removed_at - generated_at
    ids.append(id)
    frequencies[travel_time] += 1
    travel_times.append(travel_time)
    # print("travel_time", travel_time, id)
    return


def bridge_record(number, time_waited):
    bridge_waited_time[number] += time_waited
    return


def get_bridge_waited_time():
    return bridge_waited_time


def write_to_file_and_return(scenario):
    ids.append("average")
    travel_times.append(statistics.mean(travel_times))
    d = {"ID": ids, "travel_time": travel_times}
    df = pd.DataFrame(data=d)

    df.to_csv(
        os.path.join(BASE_DIR, "experiment", "scenario" + str(scenario) + ".csv"),
        index=False,
    )
    return ids, travel_times, frequencies
