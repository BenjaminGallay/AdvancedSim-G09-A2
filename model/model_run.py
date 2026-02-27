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
    [0, 0, 0, 0.05],
    [0, 0, 0, 0.1],
    [0, 0, 0.05, 0.1],
    [0, 0, 0.1, 0.2],
    [0, 0.05, 0.1, 0.2],
    [0, 0.1, 0.2, 0.4],
    [0.05, 0.1, 0.2, 0.4],
    [0.1, 0.2, 0.4, 0.8],
]


# run time 1000 ticks
run_length = 10000

seed = 1234567

sim_model = BangladeshModel(
    seed=seed, breakdown_probabilities=BREAKDOWN_PROBABILITIES[0]
)

# Check if the seed is set
print("SEED " + str(sim_model._seed))

# One run with given steps
for i in range(run_length):
    sim_model.step()
recorder.write_to_file()
