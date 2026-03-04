import statistics
import time

import matplotlib.pyplot as plt
import numpy
import statistical_recorder

from model import BangladeshModel, analytical_recorder

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

choice_dict = {
    "0": [0],
    "1": [1],
    "2": [2],
    "3": [3],
    "4": [4],
    "5": [5],
    "6": [6],
    "7": [7],
    "8": [8],
    "all": range(9),
    "bonus": range(1, 8),
}

# scenario 0 = no bridges breaking down : baseline travel time. scenario 8 = most likely breakdowns
# run time 7200 ticks = 5*24h runtime
run_length = 7200
number_of_seeds = 10
rand = int(1000000000 * time.time() % 1000000)
seeds = range(rand, rand + number_of_seeds)


def get_choice():
    valid_choice = False
    bonus = False
    print("Select an option :")
    print("- number 0 to 8, runs the corresponding scenario")
    print("- 'all' runs all scenarios successively")
    print("- 'bonus' runs scenarios 1 to 7 to answer the bonus exercise")

    while not valid_choice:
        choice = input("Enter your choice : ")
        if choice in choice_dict:
            valid_choice = True
            if choice == "bonus":
                bonus = True
        else:
            print("invalid input, please try again")

    return choice_dict[choice], bonus


scenarios, bonus = get_choice()
# Loop through all scenarios
for scenario in scenarios:
    statistical_recorder.reset_times()
    print(f"\n--- Running scenario {scenario} ---")

    for seed in seeds:
        analytical_recorder.reset()
        sim_model = BangladeshModel(
            seed=seed, breakdown_probabilities=BREAKDOWN_PROBABILITIES[scenario]
        )

        # Check if the seed is set
        print("SEED " + str(sim_model._seed))

        # One run with given steps
        for i in range(run_length):
            sim_model.step()

    ids, travel_times, frequencies = statistical_recorder.write_to_file_and_return(
        scenario
    )
    print(
        "statistical average travel time for scenario",
        scenario,
        ":",
        statistics.mean(travel_times),
    )
    print(
        "analytical expected travel time for scenario",
        scenario,
        ":",
        analytical_recorder.get_expected_mean_travel_time(),
    )
    if scenario == 7 and bonus:
        bridge_waited_time = statistical_recorder.get_bridge_waited_time()
        worst_indexes = numpy.argsort(bridge_waited_time)[::-1][:5]
        waited_times = [bridge_waited_time[i] for i in worst_indexes]
        print(
            "\nBonus question --- Here are the bridges that have caused the most combined time delay"
        )
        for i in range(5):
            print(
                f"{i + 1}. Bridge {1000000 + worst_indexes[i]}, which caused a combined delay of {waited_times[i]} ticks."
            )
