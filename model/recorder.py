import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ids = []
travel_times = []


def truck_record(id, generated_at, removed_at):
    travel_time = removed_at - generated_at
    ids.append(id)
    travel_times.append(travel_time)
    print("travel_time", travel_time, id)
    return


def write_to_file():
    d = {"ID": ids, "travel_time": travel_times}
    df = pd.DataFrame(data=d)

    df.to_csv(os.path.join(BASE_DIR, "experiment", "scenario0.csv"), index=False)
