"""
A main script to run any sample code (in order to avoid modification of the sample in the loss.py)
This is considered a temporary file and is not essential to the framework.
"""
from loss import Loss
from pathlib import Path


# ---- Definition of inputs
# Main directory
directory = Path.cwd().parents[0] / ".applications/case1"
# Replacement costs
replCost = 1376758.0
# Demolition function
demolition = {"median": 0.015 * 100, "cov": 0.30}
# Path to input files
nrhaFileName = directory / "Output1/RCMRF" / "ida_processed.pickle"
hazardFileName = directory / "Hazard-LAquila-Soil-C.pkl"
rsFileName = directory / "Output1" / "RS.pickle"
slfFileName = directory / "Output1/Cache" / "SLFs.pickle"
outputDir = directory / "Output1/Cache"
period = 0.97

# Initializing Loss object
l = Loss(calculate_pga_values=True, nrhaFileName=nrhaFileName, hazardFileName=hazardFileName, include_demolition=True,
         non_directional_factor=1.0, collapse=None, demolition=demolition, replCost=replCost, performSimulations=False,
         rsFileName=rsFileName, period=period, iml_range_consistent=False, slfFileName=slfFileName)
# Get start time
start_time = l.get_init_time()
# Read the inputs of the framework
inputs = l.read_input()
# Calculate losses
loss = l.calc_losses(inputs["NRHA"], inputs["residuals"])
# Calculate loss ratios
E_int = l.loss_ratios(loss, demolition_threshold=1.0)
# Calculate EAL
eal, cache = l.get_eal(E_int["E_LT"], inputs["Hazard"])
print(f"[EAL]: {eal: .2f}%")

# Get running time
l.get_time(start_time)

import pickle

with open(outputDir/"cacheLosses.pickle", "wb") as f:
    pickle.dump(cache, f, pickle.HIGHEST_PROTOCOL)

loss.to_csv(outputDir/"loss.csv", index=False)


