import statistics

import matplotlib.pyplot as plt
import numpy
import recorder

from model import BangladeshModel

"""
    Run simulation
    Print output at terminal
"""

# ---------------------------------------------------------------

# run time 5 x 24 hours; 1 tick 1 minute
# run_length = 5 * 24 * 60

BREAKDOWN_PROBABILITIES = [
    [0, 0, 0, 0],
    [0, 0, 0, 0.05],
    [0, 0, 0, 0.1],
    [0, 0, 0.05, 0.1],
    [0, 0, 0.1, 0.2],
    [0, 0.05, 0.1, 0.2],
    [0, 0.1, 0.2, 0.4],
    [0.05, 0.1, 0.2, 0.4],
    [0.1, 0.2, 0.4, 0.8],
]

# scenario 0 = no bridges breaking down : baseline travel time. scenario 8 = most likely breakdowns
# run time 7200 ticks = 5*24h runtime
run_length = 7200
number_of_seeds = 10
seeds = range(100, 100 + number_of_seeds)

# Loop through all scenarios
for scenario in range(len(BREAKDOWN_PROBABILITIES)):
    recorder.reset_times()
    print(f"\n--- Running scenario {scenario} ---")

    for seed in seeds:
        sim_model = BangladeshModel(
            seed=seed, breakdown_probabilities=BREAKDOWN_PROBABILITIES[scenario]
        )

        # Check if the seed is set
        print("SEED " + str(sim_model._seed))

        # One run with given steps
        for i in range(run_length):
            sim_model.step()

    ids, travel_times, frequencies = recorder.write_to_file_and_return(scenario)
    print(
        "average travel time for scenario", scenario, ":", statistics.mean(travel_times)
    )
    if scenario == 7:
        bridge_waited_time = recorder.get_bridge_waited_time()
        print(
            "worst bridges are :",
            numpy.argsort(bridge_waited_time)[-5:],
            "where trucks have waited :",
            [bridge_waited_time[i] for i in numpy.argsort(bridge_waited_time)[-5:]],
        )
