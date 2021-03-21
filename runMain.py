"""
A main script to run any sample code (in order to avoid modification of the sample in the loss.py)
This is considered a temporary file and is not essential to the framework.
"""
from loss import Loss
from pathlib import Path
import numpy as np
import pickle
from tools.visualize import Visualize
import sys


# ---- Definition of inputs
# Main directory
directory = Path.cwd().parents[0] / ".applications/compdyn2021"
# Replacement costs
replCost = 1213.4 * 432 * 3
# Demolition function
demolition = {"median": 0.015 * 100, "cov": 0.30}
# Path to input files
nrhaFileName = directory / "RCMRF" / "ida_processed.pickle"
hazardFileName = directory / "Hazard-LAquila-Soil-C.pkl"
rsFileName = directory / "RS.pickle"
slfFileName = directory / "slfs.pickle"
outputDir = directory
period = np.power(1.09 * 1.05, 0.5)

# Initializing Loss object
l = Loss(calculate_pga_values=True, nrhaFileName=nrhaFileName, hazardFileName=hazardFileName, include_demolition=True,
         non_directional_factor=1.2, collapse=None, demolition=demolition, replCost=replCost, performSimulations=False,
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

with open(outputDir/"cacheLosses.pickle", "wb") as f:
    pickle.dump(cache, f, pickle.HIGHEST_PROTOCOL)

loss.to_csv(outputDir/"loss.csv", index=False)

# # Data visualization
# # EAL visualization
# sflag = False
# pflag = False
# v = Visualize()
# cache_eal = v.plot_eal(cache, loss, pflag=pflag, sflag=sflag, replCost=replCost)
# # Loss curves, vulnerability curves
# v.plot_loss_curves(loss, pflag=pflag, sflag=sflag)
# # Plot vulnerability curve
# v.plot_vulnerability(cache, demolition_threshold=0.6, pflag=pflag, sflag=sflag)
# # Area plots of loss contributions
# v.area_plots(cache, loss, pflag=pflag, sflag=sflag)


