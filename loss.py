import os
import timeit
from pathlib import Path
import numpy as np
import pickle
import pandas as pd
from scipy.interpolate import interp1d
from tools.cost import Cost
from tools.sat1 import SaT1
from tools.visualize import Visualize


# TODO, add loss curve, EAL, PV estimations
class Loss:
    def __init__(self, directory=None, idaFileName=None, rsFileName=None, periodsFileName=None, slfFileName=None,
                 hazardFileName=None, iml_max=2.0, iml_step=0.05, calculate_pga_values=False, include_demolition=True,
                 slf_normalization=False, non_directional_factor=1.0, collapse=None, use_beta_MDL=0.15,
                 demolition=None, replCost=1):
        """
        Initialize Loss estimation framework based on story-loss functions
        :param directory: str                           Main working directory
        :param idaFileName: str                         Incremental dynamic analysis (IDA) filename
        :param rsFileName: str                          Scaled Response spectra (RS) file name
        :param periodsFileName: str                     File name containing 1st mode periods of the buildings
        :param slfFileName: str                         SLF file name
        :param hazardFileName: str                      Hazard function file name
        :param iml_max: float                           Max IML
        :param iml_step: float                          IML step
        :param calculate_pga_values: bool               Whether to calculate the PGA values (usually True if none
                                                        provided)
        :param include_demolition: bool                 Whether to calculate loss contribution from demolition
        :param slf_normalization: bool                  Whether to perform SLF normalization based on provided replacement cost or not
        :param non_directional_factor: float            Non-directional conversion factor for components of no-directionality
        :param collapse: dict                           Median and dispersion of collapse fragility function, or None if needs to be computed
        :param use_beta_MDL: float                      Standard deviation accounting for modelling uncertainty to use, if default calculation is opted
        :param demolition: dict                         Median and COV of demolition fragility function
        :param replCost: float                          Replacement cost of the building
        """
        # Set the main directory
        if directory is None:
            self.directory = Path.cwd()
        else:
            self.directory = directory

        # Get the ida file name
        self.idaFileName = idaFileName

        # Get the RS file name
        self.rsFileName = rsFileName

        # Get the periods file name
        self.periodsFileName = periodsFileName

        # Get the SLF file name
        self.slfFileName = slfFileName

        # Get the Hazard file name
        self.hazardFileName = hazardFileName

        # Get the intensity measure level (IML) range
        self.iml_range = np.arange(iml_step, iml_max + iml_step, iml_step)

        # Calculation of PGA values
        self.calculate_pga_values = calculate_pga_values

        # Demolition contribution flag
        self.include_demolition = include_demolition
        
        # SLF normalization flag
        self.slf_normalization = slf_normalization
        
        # Non-directional conversion factor
        self.non_directional_factor = non_directional_factor

        # Initialize number of stories
        self.nstories = None

        # Collapse CDF parameters
        self.collapse = collapse

        # Modelling uncertainty for collapse probability calculation
        self.use_beta_MDL = use_beta_MDL

        # Demolition CDF parameters
        self.demolition = demolition
        
        # Replacement cost of the building
        self.replCost = replCost

    @staticmethod
    def get_init_time():
        """
        Records initial time
        :return: float                      Initial time
        """
        start_time = timeit.default_timer()
        return start_time

    @staticmethod
    def truncate(n, decimals=0):
        """
        Truncates time with given decimal points
        :param n: float                     Time
        :param decimals: int                Decimal points
        :return: float                      Truncated time
        """
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier

    def get_time(self, start_time):
        """
        Prints running time in seconds and minutes
        :param start_time: float            Initial time
        :return: None
        """
        elapsed = timeit.default_timer() - start_time
        print('Running time: ', self.truncate(elapsed, 1), ' seconds')
        print('Running time: ', self.truncate(elapsed / float(60), 2), ' minutes')

    def read_input(self):
        """
        Reads the inputs for the loss assessment
        IDA: summary_results:   IM is in [g]
                                maxFA is in [g]
                                maxISDR
                                maxRISDR
        IDA: IDA:               IM is in [g]
                                ISDR is in [%]
                                PFA is in [g]
                                RISDR
        RS is in s vs g
        :return: dict                                   Dictionary containing IDA, RS, Periods and Nstories information
        """
        # Get client directory
        clientDirectory = self.directory / "client"

        """ Read IDA outputs if file name was not provided, search for any within the client directory
        This will run all IDA files, so be careful as it could enter an extensive analysis requiring a large
        computational effort
        It is highly recommended to provide the file names MANUALLY"""
        if self.idaFileName is None:
            for file in os.listdir(clientDirectory):
                if (file.endswith(".pickle") or file.endswith(".pkl")) and file.startswith("ida_"):
                    ida_file = open(clientDirectory / file, "rb")
                    ida_outputs = pickle.load(ida_file)
                    ida_file.close()
                    self.idaFileName = clientDirectory / file
        else:
            ida_outputs = {}
            for fileName in self.idaFileName:
                ida_file = open(clientDirectory / fileName, "rb")
                ida_temp = pickle.load(ida_file)
                ida_file.close()
                fileName = fileName.strip(".pickle")
                ida_outputs[fileName[-1]] = ida_temp

        # Reading of the RS file
        if self.rsFileName is None:
            for file in os.listdir(clientDirectory):
                if file.startswith("RS") or file.startswith("rs"):
                    rs = pd.read_pickle(clientDirectory / file)
                    self.rsFileName = clientDirectory / file
        else:
            rs = pd.read_pickle(clientDirectory / self.rsFileName)

        # Get the fundamental (1st mode) periods of the buildings under consideration
        if self.periodsFileName is None:
            for file in os.listdir(clientDirectory):
                if file.startswith("period") or file.startswith("Period"):
                    periods = pd.read_csv(clientDirectory / file, index_col=None)
                    self.periodsFileName = clientDirectory / file
        else:
            periods = pd.read_csv(clientDirectory / self.periodsFileName, index_col=None)

        # Get the hazard function
        if self.hazardFileName is None:
            for file in os.listdir(clientDirectory):
                if "hazard" in file or "Hazard" in file:
                    hazard = pd.read_csv(clientDirectory / file)
                    self.hazardFileName = clientDirectory / file
        else:
            hazard = pd.read_csv(clientDirectory / self.hazardFileName)

        # Get number of stories of the building
        for i in ida_outputs["x"]["summary_results"]:
            for j in ida_outputs["x"]["summary_results"][i].keys():
                for k in ida_outputs["x"]["summary_results"][i][j].keys():
                    for key in ida_outputs["x"]["summary_results"][i][j].get(k):
                        self.nstories = key
                    break
                break
            break

        # TODO, add batch mode, where multiple buildings will be analyzed
        # Check whether peak ground acceleration (PGA) values are provided, if not, calculate them
        if self.calculate_pga_values:
            ida_outputs = self.calc_PGA()

        return {"IDA": ida_outputs, "RS": rs, "Periods": periods, "Hazard": hazard, "Nstories": self.nstories}

    def calc_PGA(self):
        """
        Calculate the PGA values
        :return: dict                               IDA outputs
        """
        sat1 = SaT1(self.rsFileName, self.periodsFileName, self.idaFileName)
        outputs = sat1.calc_ida_PGA()
        return outputs

    def calc_losses(self, ida_outputs):
        """
        SLF:                                        IDR
                                                    PFA is in [g]
        Calculates losses based on SLFs
        :param ida_outputs: dict                    IDA outputs
        :return: dict                               Computed, disaggregated and total losses
        """
        # TODO, add replacement cost for SLF normalization and perform it here, rather than providing already
        # normalized SLFs
        # Get client directory
        clientDirectory = self.directory / "client"
            
        if self.slfFileName is None:
            for file in os.listdir(clientDirectory):
                if file.startswith("slf"):
                    cost = Cost(slf_filename=clientDirectory / file, include_demolition=self.include_demolition,
                                nonDirFactor=self.non_directional_factor)
            self.slfFileName = clientDirectory / file
        else:
            cost = Cost(slf_filename=clientDirectory / self.slfFileName, include_demolition=self.include_demolition,
                        nonDirFactor=self.non_directional_factor)
        losses = cost.calc_losses(ida_outputs, self.iml_range, self.nstories, collapse=self.collapse,
                                  use_beta_MDL=self.use_beta_MDL, demolition=self.demolition, replCost=self.replCost)
        return losses
    
    def loss_ratios(self, losses, demolition_threshold=0.6):
        """
        Gets interpolation functions for the loss curves and applies the demolition threshold
        :param losses: dict                         Losses as a ratio of the total replacement cost
        :param demolition_threshold: float          Threshold beyond which the building is to be in complete loss
        :return: dict                               Interpolation functions as a cache for later use
        """
        # Get the IML range for plotting
        IML = np.insert(np.array([float(i) for i in losses.index]), 0, 0)

        # Concatenating a zero to the beginning of the loss arrays to avoid interpolation issues and for visualization
        # IDR sensitive structural components, no collapse, no demolition
        E_NC_ND_ISDR_S = np.insert(np.array(losses['E_NC_ND_ISDR_S']), 0, 0)
        # IDR sensitive non-structural components, no collapse, no demolition
        E_NC_ND_ISDR_NS = np.insert(np.array(losses['E_NC_ND_ISDR_NS']), 0, 0)
        # IDR sensitive components, no collapse, no demolition
        E_NC_ND_ISDR_TOTAL = np.insert(np.array(losses['E_NC_ND_ISDR_TOTAL']), 0, 0)
        # PFA sensitive components, no collapse, no demolition
        E_NC_ND_PFA_TOTAL = np.insert(np.array(losses['E_NC_ND_PFA_TOTAL']), 0, 0)
        # Structural components, no collapse, no demolition
        E_NC_ND_S = np.insert(np.array(losses['E_NC_ND_S']), 0, 0)
        # Non-structural components, no collapse, no demolition
        E_NC_ND_NS = np.insert(np.array(losses['E_NC_ND_NS']), 0, 0)
        # No collapse, no demolition
        E_NC_ND = np.insert(np.array(losses['E_NC_ND']), 0, 0)
        # No collapse, demolition
        E_NC_D = np.insert(np.array(losses['E_NC_D']), 0, 0)
        # Collapse
        E_C = np.insert(np.array(losses['E_C']), 0, 0)
        # Total
        E_LT = np.insert(np.array(losses['E_LT']), 0, 0)
        # Applying the demolition threshold, beyond which complete loss is assumed
        E_LT[E_LT >= demolition_threshold*self.replCost] = self.replCost
        
        # Interpolation functions
        E_NC_ND_ISDR_S_spline = interp1d(IML, E_NC_ND_ISDR_S)
        E_NC_ND_ISDR_NS_spline = interp1d(IML, E_NC_ND_ISDR_NS)
        E_NC_ND_ISDR_TOTAL_spline = interp1d(IML, E_NC_ND_ISDR_TOTAL)
        E_NC_ND_PFA_TOTAL_spline = interp1d(IML, E_NC_ND_PFA_TOTAL)
        E_NC_ND_S_spline = interp1d(IML, E_NC_ND_S)
        E_NC_ND_NS_spline = interp1d(IML, E_NC_ND_NS)
        E_NC_ND_spline = interp1d(IML, E_NC_ND)
        E_NC_D_spline = interp1d(IML, E_NC_D)
        E_C_spline = interp1d(IML, E_C)
        E_LT_spline = interp1d(IML, E_LT)

        # Into a dictionary
        E_interpolation_functions = {}
        for key in losses.keys():
            exp_loss = np.insert(np.array(losses[key]), 0, 0)
            if key == "E_LT":
                exp_loss[exp_loss >= demolition_threshold*self.replCost] = self.replCost
            E_interpolation_functions[key] = interp1d(IML, exp_loss)

        return E_interpolation_functions

    def get_eal(self, spline, hazard, method="Porter"):
        """
        Computation of EAL
        :param spline: interp1d             1D interpolation function for the expected total loss
        :param hazard: dict                 Hazard function
        :param method: str                  Calculation method: Porter -> Method 1, other -> applies numpy (technically the same)
        :return: float                      EAL as a % of the total replacement cost
        :return cache: dict                 EAL bins, IML range and MAF for Visualization
        """
        # IML step of hazard function
        iml_hazard = np.array(hazard["Sa(T1)"])
        # Ground shaking Mean annual frequency of exceedance
        probs = np.array(hazard["MAFE"])
        
        # Calling the Cost object
        c = Cost()
        if method == "Porter":
            # Loss as the ratio of the replacement cost
            mdf = spline(iml_hazard)/self.replCost
            #  Computing the EAL ratio in %
            eal, cache = c.compute_eal(iml_hazard, probs, mdf, rc=1, method=method)
        else:
            # IML
            interpolation = interp1d(iml_hazard, probs)
            iml = np.arange(iml_hazard[0], iml_hazard[-1], 0.01)
            l = interpolation(iml)
            # Loss as the ratio of the replacement cost
            mdf = spline(iml)/self.replCost
            #  Computing the EAL ratio in %
            eal, cache = c.compute_eal(iml, probs, mdf, rc=1, method=method)
        return eal, cache


if __name__ == "__main__":

    # TODO, rsFileName being used when? Update file
    #  slf_normalization being used when?
    # Inputs
    collapse = {"theta": 1.49, "beta": 0.33}
    demolition = {"median": 0.015, "cov": 0.30}
    replCost = 3929937.
    # Set up a main directory
    directory = Path.cwd()
    # Loss object Initialization
    # SLFs are already normalized with respect to the building replacement cost
    l = Loss(calculate_pga_values=False, idaFileName=["ancona_x.pickle", "ancona_y.pickle"], rsFileName="RS.pickle",
             hazardFileName="ancona_hazard.csv", periodsFileName="Periods.tcl", slfFileName="slfsVersion6.pickle",
             include_demolition=True, non_directional_factor=1.2, collapse=collapse, demolition=demolition, replCost=replCost)
    # Get start time
    start_time = l.get_init_time()

    # Read the inputs of the framework
    inputs = l.read_input()
    # Calculate losses
    loss = l.calc_losses(inputs["IDA"])
    # Calculate loss ratios
    E_int = l.loss_ratios(loss, demolition_threshold=0.6)
    # Calculate EAL
    eal, cache = l.get_eal(E_int["E_LT"], inputs["Hazard"])
    print(f"[EAL]: {eal: .2f}%")
    # Data visualization
    # EAL visualization
    sflag = False
    pflag = True
    v = Visualize()
    cache_eal = v.plot_eal(cache, loss, pflag=pflag, sflag=sflag, replCost=replCost)
    # Loss curves, vulnerability curves
    v.plot_loss_curves(loss, pflag=pflag, sflag=sflag)
    # Plot vulnerability curve
    v.plot_vulnerability(cache, demolition_threshold=0.6, pflag=pflag, sflag=sflag)
    # Area plots of loss contributions
    v.area_plots(cache, loss, pflag=pflag, sflag=sflag)

    # Get running time
    l.get_time(start_time)
    
    
    # import pickle
    with open("cache.pkl", "wb") as f:
        pickle.dump(cache, f, pickle.HIGHEST_PROTOCOL)
        
    with open("cache_eal.pkl", "wb") as f:
        pickle.dump(cache_eal, f, pickle.HIGHEST_PROTOCOL)
    
    with open("loss.pkl", "wb") as f:
        pickle.dump(loss, f, pickle.HIGHEST_PROTOCOL)
        
        
        
        
        
        
    
    
    
