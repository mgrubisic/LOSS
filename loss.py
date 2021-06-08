import os
import timeit
from pathlib import Path
import numpy as np
import pickle
import pandas as pd
from scipy.interpolate import interp1d
from pyDOE import lhs
from scipy.stats.distributions import norm
from tools.cost import Cost
from tools.sat1 import SaT1
from tools.visualize import Visualize


# TODO, add loss curve, PV estimations
class Loss:
    def __init__(self, directory=None, nrhaFileName=None, rsFileName=None, period=None, slfFileName=None,
                 hazardFileName=None, calculate_pga_values=False, include_demolition=True, non_directional_factor=1.0,
                 collapse=None, use_beta_MDL=0.15, demolition=None, replCost=1.0, betas=None, performSimulations=False,
                 num_realization=1000, iml_range_consistent=False, flag3d=True, normalize=False):
        """
        Initialize Loss estimation framework based on story-loss functions
        :param directory: str                           Main working directory
        :param nrhaFileName: str                        Incremental dynamic analysis (IDA) filename
        :param rsFileName: str                          Scaled Response spectra (RS) file name
        :param period: float                            Fundamental period of the structure
        :param slfFileName: str                         SLF file name
        :param hazardFileName: str                      Hazard function file name
        :param calculate_pga_values: bool               Whether to calculate the PGA values (usually True if none
                                                        provided)
        :param include_demolition: bool                 Whether to calculate loss contribution from demolition
        :param non_directional_factor: float            Non-directional conversion factor for components of
                                                        no-directionality
        :param collapse: dict                           Median and dispersion of collapse fragility function, or None if
                                                        needs to be computed
        :param use_beta_MDL: float                      Standard deviation accounting for modelling uncertainty to use,
                                                        if default calculation is opted
        :param demolition: dict                         Median and COV of demolition fragility function
        :param replCost: float                          Replacement cost of the building
        :param betas: ndarray                           Modelling uncertainties (epistemic variabilities)
        :param performSimulations: bool                 Whether to perform simulations using Latin Hypercube sampling
        :param num_realization: int                     Number of realizations
        :param iml_range_consistent: bool               If True derives IML range (same for each record), else uses IML
                                                        range different for each record
        :param flag3d: bool                             3D or 2D modelling
        :param normalize: bool                          Normalize SLFs by replacement cost
        """
        # Set the main directory
        if directory is None:
            self.directory = Path.cwd()
        else:
            self.directory = directory

        # Get the ida file name
        self.nrhaFileName = nrhaFileName

        # Get the RS file name
        self.rsFileName = rsFileName

        # Get the periods file name
        self.period = period

        # Get the SLF file name
        self.slfFileName = slfFileName

        # Get the Hazard file name
        self.hazardFileName = hazardFileName

        # Calculation of PGA values
        self.calculate_pga_values = calculate_pga_values

        # Demolition contribution flag
        self.include_demolition = include_demolition

        # Non-directional conversion factor
        self.non_directional_factor = non_directional_factor

        # Initialize number of stories
        self.n_stories = None

        # Collapse CDF parameters
        self.collapse = collapse

        # Modelling uncertainty for collapse probability calculation
        self.use_beta_MDL = use_beta_MDL

        # Demolition CDF parameters
        self.demolition = demolition

        # Replacement cost of the building
        self.replCost = replCost

        # Modelling uncertainties
        self.betas = betas

        # Number of realizations for simulated demands
        self.num_realization = num_realization

        # Simulations
        self.performSimulations = performSimulations

        # IML range
        self.iml_range = None

        # IML consistency
        self.iml_range_consistent = iml_range_consistent

        # Mode of modelling
        self.flag3d = flag3d

        # Normalization flag
        self.normalize = normalize

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

    def calc_PGA(self, idaOutputs):
        """
        Calculate the PGA values
        :param idaOutputs: dict                     IDA outputs
        :return: dict                               Modified IDA outputs (inclusion of PGA)
        """
        sat1 = SaT1(self.rsFileName, self.period, idaOutputs)
        outputs = sat1.calc_ida_PGA()
        return outputs

    def _into_ndarray(self, data):
        """
        Transforms a dictionary of NRHA outputs into an ndarray
        :param data: dict                   Dictionary of NRHA outputs
        :return: ndarray                    Ndarray of NRHA outputs
        """
        # Get the dimensions of the desired ndarray
        ngm = len(data)
        nvar = 0
        niml = 0
        for key in data.keys():
            niml = len(data[key])
            for iml in data[key].keys():
                for edp in data[key][iml].keys():
                    # Ignore residual drifts
                    if edp != "maxRISDR":
                        nvar += len(data[key][iml][edp])
                break
            break

        # Check if modelling uncertainties were provided
        if self.betas is None:
            # If not provided, set all uncertainties to zero
            self.betas = np.zeros((niml, 1))

        # Initialize NRHA outputs ndarray variable
        nrha = np.zeros((niml, ngm, nvar))

        # Initialize IDs / Counts
        gm_cnt = 0
        # Iterate for each GM
        for gm in data.keys():
            # Iterate for each IM level
            iml_cnt = 0
            for iml in data[gm].keys():
                var_cnt = 0
                # Iterate for each edp type, except for residual drifts
                for edp in data[gm][iml].keys():
                    if edp != "maxRISDR":
                        # Iterate for each variable
                        for var in data[gm][iml][edp]:
                            # Set to nan if 0 was recorded (collapsed cases
                            value = data[gm][iml][edp][var] if data[gm][iml][edp][var] != 0.0 else np.nan
                            nrha[iml_cnt][gm_cnt][var_cnt] = value
                            # Increase variable ID
                            var_cnt += 1
                # Increase IML ID
                iml_cnt += 1
            # Increase gm ID
            gm_cnt += 1
        return nrha

    def simulate_demands(self, demands, beta):
        """
        Latin Hypercube sampling to calculate the distribution of probable losses for each ground motion intensity or
        earthquake scenario. The original demands of shape (ngm, nvar) will be transformed to simulated shape (nr, nvar)
        Where instead of N ground motions, there will be N simulations for each EDP (variable).
        :param demands: dict                NRHA demands, EDPs
        :param beta: float                  Modelling uncertainty associated with the IML
        :return: dict                       Simulated demands
        """
        ngm, nvar = demands.shape
        # Reshape uncertainties to match
        betas = np.array([beta] * nvar).reshape(nvar, 1)

        # Take natural logarithm of the EDPs
        lnEDPs = np.log(demands)

        # Find the mean matrix of lnEDPs
        lnEDPs_mean = np.mean(lnEDPs, axis=0).reshape(lnEDPs.shape[1], 1)

        # Find covariance matrix of lnEDPs
        lnEDPs_cov = np.cov(lnEDPs, rowvar=False)

        # Find the rank of covariance matrix of lnEDPs
        lnEDPs_cov_rank = np.linalg.matrix_rank(lnEDPs_cov)

        # Inflate the variances with epistemic variability
        sigma = np.array([np.sqrt(np.diag(lnEDPs_cov))]).transpose()
        sigmap2 = np.power(sigma, 2)

        sigma_t = sigma.transpose()
        R = lnEDPs_cov / (sigma * sigma_t)

        # Inflate variance for modelling uncertainty
        sigmap2 = sigmap2 + (betas * betas)
        sigma = np.sqrt(sigmap2)
        sigma2 = sigma * sigma.transpose()
        lnEDPs_cov_inflated = R * sigma2

        # Find the eigenvalues, eigenvectors of the covariance matrix
        eigenValues, eigenVectors = np.linalg.eigh(lnEDPs_cov_inflated)

        # Partition L_total to L_use. L_use is the part of eigenvector matrix
        # L_total that corresponds to positive eigenvalues
        # Similarly for D_use

        if lnEDPs_cov_rank >= nvar:
            L_use = eigenVectors
            D2_use = eigenValues
        else:
            L_use = eigenVectors[:, nvar - lnEDPs_cov_rank:]
            D2_use = eigenValues[nvar - lnEDPs_cov_rank:]

        # Find the square roof of D2_use
        D_use = np.diag(np.power(D2_use, 0.5))

        # Generate Standard random numbers
        if lnEDPs_cov_rank >= nvar:
            U = lhs(nvar, self.num_realization)
        else:
            U = lhs(lnEDPs_cov_rank, self.num_realization)
        U = norm(loc=0, scale=1).ppf(U)

        U = U.transpose()

        # Create Lambda=D_use
        L = np.matmul(L_use, D_use)

        # Create realizations matrix
        Z = np.matmul(L, U) + np.matmul(lnEDPs_mean, np.ones([1, self.num_realization]))

        lnEDPs_sim_mean = np.mean(Z, axis=1).reshape(nvar, 1)
        lnEDPs_sim_cov = np.cov(Z)

        # Values of A should be close to 1, as the means of simulated demands should be the same as the means of
        # original demands (currently imposing 5% tolerance)
        A = lnEDPs_sim_mean / lnEDPs_mean
        test = abs(A-1)*100
        if any(test[test >= 5.0]):
            print("[WARNING] Means of simulated demands are not equal to the means of original demands!")

        B = lnEDPs_sim_cov / lnEDPs_cov
        W = np.exp(Z).transpose()
        return W

    def get_residuals(self, demands, sorting=False):
        """
        Transforms residual drifts into an array format
        :param demands: dict                    NRHA outputs
        :param sorting: bool                    Whether to sort the RIDR based on IML
        :return: arraay                         Residual drifts
        """
        key = next(iter(demands))
        ridr = np.zeros((len(demands[key]["IM"]), len(demands)))
        cnt = 0
        for gm in demands:
            data = demands[gm]["RISDR"]

            if sorting:
                order = np.argsort(demands[gm]["IM"])
                data = data[order]
            ridr[:, cnt] = data
            cnt += 1
        return ridr

    def read_input(self):
        """
        NOTE: The keys will vary for other input files (might be MSA instead of IDA)
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
        # Get client directory (in case no name for the files were provided)
        clientDirectory = self.directory / "client"

        """ Read IDA outputs if file name was not provided, search for any within the client directory
        This will run all IDA files, so be careful as it could enter an extensive analysis requiring a large
        computational effort
        It is highly recommended to provide the file names MANUALLY"""
        nrhaTemp = {}
        if self.nrhaFileName is None:
            for file in os.listdir(clientDirectory):
                if (file.endswith(".pickle") or file.endswith(".pkl")) and file.startswith("ida_"):
                    nrha_file = open(clientDirectory / file, "rb")
                    nrhaTemp = pickle.load(nrha_file)
                    nrha_file.close()
                    self.nrhaFileName = clientDirectory / file
        else:
            if isinstance(self.nrhaFileName, list):
                for fileName in self.nrhaFileName:
                    nrha_file = open(fileName, "rb")
                    ida_temp = pickle.load(nrha_file)
                    nrha_file.close()
                    fileName = os.path.basename(fileName)
                    fileName = fileName.strip(".pickle")
                    nrhaTemp[fileName[-1]] = ida_temp
            else:
                nrha_file = open(self.nrhaFileName, "rb")
                ida_temp = pickle.load(nrha_file)
                nrha_file.close()
                if len(ida_temp.keys()) > 1:
                    # 3D model is being considered within a single file
                    for key in ida_temp.keys():
                        nrhaTemp[key] = ida_temp[key]
                else:
                    try:
                        nrhaTemp["IDAs"] = ida_temp[0]
                    except:
                        nrhaTemp["IDAs"] = ida_temp

        # Reading of the RS file
        rs = None
        if self.rsFileName is None:
            for file in os.listdir(clientDirectory):
                if file.startswith("RS") or file.startswith("rs"):
                    rs = pd.read_pickle(clientDirectory / file)
                    self.rsFileName = clientDirectory / file
        else:
            rs = pd.read_pickle(self.rsFileName)

        # Get the hazard function
        hazard = None
        if self.hazardFileName is None:
            for file in os.listdir(clientDirectory):
                if "hazard" in file or "Hazard" in file:
                    hazard = pd.read_csv(clientDirectory / file)
                    self.hazardFileName = clientDirectory / file
        else:
            basename = os.path.basename(self.hazardFileName)
            if basename.endswith(".csv"):
                hazard = pd.read_csv(self.hazardFileName)
            else:
                # Pickle file
                with open(self.hazardFileName, "rb") as file:
                    hazard = pickle.load(file)
                    
        # Get number of stories of the building
        key = next(iter(nrhaTemp))
        for i in nrhaTemp[key]["summary_results"]:
            for j in nrhaTemp[key]["summary_results"][i].keys():
                for k in nrhaTemp[key]["summary_results"][i][j].keys():
                    for m in nrhaTemp[key]["summary_results"][i][j].get(k):
                        self.n_stories = m
                    if k != "maxFA":
                        # Sometimes maxFA key might not include the PGA value, hence stories might be read incorrectly
                        break
                break
            break
        # Check whether peak ground acceleration (PGA) values are provided, if not, calculate them
        if self.calculate_pga_values:
            nrhaTemp = self.calc_PGA(nrhaTemp)

        # Calculate original IML range (this assumes that each record has the same iml range
        sortingRIDR = False
        if self.iml_range_consistent:
            for gm in nrhaTemp[key]["IDA"].keys():
                self.iml_range = nrhaTemp[key]["IDA"][gm]["IM"]
                break
        else:
            # Get ndarray of IML range for each record
            keygm = next(iter(nrhaTemp[key]["IDA"]))

            self.iml_range = np.zeros((len(nrhaTemp[key]["IDA"][keygm]["IM"]), len(nrhaTemp[key]["IDA"])))

            cnt = 0
            for rec in nrhaTemp[key]["IDA"].keys():
                # "IDA" is not sorted, while "summary_results" are sorted, and since EDPs will be read from
                # "summary_results", we need to sort the IM values here
                self.iml_range[:, cnt] = np.sort(nrhaTemp[key]["IDA"][rec]["IM"])
                sortingRIDR = True
                cnt += 1

        # Modify NRHA dict into a ndarray
        nrhaOutputs = {}
        for key in nrhaTemp.keys():
            nrhaOutputs[key] = self._into_ndarray(nrhaTemp[key]["summary_results"])

        # Replace nans with means of records at each IML
        for key in nrhaOutputs:
            means_nrha = np.nanmean(nrhaOutputs[key], axis=1)
            nrhaOutputs[key] = np.moveaxis(nrhaOutputs[key], 1, 0)
            nrhaOutputs[key] = np.where(np.isnan(nrhaOutputs[key]), means_nrha, nrhaOutputs[key])
            nrhaOutputs[key] = np.moveaxis(nrhaOutputs[key], 0, 1)

        # Simulated outputs
        if self.performSimulations:
            nrha = {}
            for d in nrhaOutputs.keys():
                nrha[d] = np.zeros((nrhaOutputs[d].shape[0], self.num_realization, nrhaOutputs[d].shape[2]))
                for i in range(nrhaOutputs[d].shape[0]):
                    nrha[d][i, :, :] = self.simulate_demands(nrhaOutputs[d][i, :, :], self.betas[i])
        else:
            nrha = nrhaOutputs

        with open(self.nrhaFileName.parents[0] / "nrhaCache.pickle", "wb") as f:
            pickle.dump(nrha, f, pickle.HIGHEST_PROTOCOL)

        # Get residual drifts (based only on one direction)
        ridr = self.get_residuals(nrhaTemp[key]["IDA"], sorting=sortingRIDR)

        return {"NRHA": nrha, "residuals": ridr, "RS": rs, "Hazard": hazard, "Nstories": self.n_stories}

    def calc_losses(self, nrhaOutputs, ridr):
        """
        SLF:                                        IDR
                                                    PFA is in [g]
        Calculates losses based on SLFs
        :param nrhaOutputs: dict                    NRHA outputs (two 3D arrays)
        :param ridr: array                          Residual drifts, make sure to provide consistent outputs with
                                                    demolition fragility function
        :return: dict                               Computed, disaggregated and total losses
        """
        cost = None
        if self.slfFileName is None:
            # Get client directory
            clientDirectory = self.directory / "client"
            for file in os.listdir(clientDirectory):
                if file.startswith("slf") or file.startswith("SLF"):
                    cost = Cost(self.n_stories, slf_filename=clientDirectory / file,
                                include_demolition=self.include_demolition, nonDirFactor=self.non_directional_factor)
                    self.slfFileName = clientDirectory / file
                else:
                    raise ValueError("[EXCEPTION] SLFs are missing!")
        else:
            cost = Cost(self.n_stories, slf_filename=self.slfFileName, include_demolition=self.include_demolition,
                        nonDirFactor=self.non_directional_factor)
        
        losses = cost.calc_losses(nrhaOutputs, ridr, self.iml_range, collapse=self.collapse, flag3d=self.flag3d,
                                  use_beta_mdl=self.use_beta_MDL, demolition=self.demolition, repl_cost=self.replCost,
                                  normalize=self.normalize)
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

        # Into a dictionary
        E_interpolation_functions = {}
        for key in losses.keys():
            exp_loss = np.insert(np.array(losses[key]), 0, 0)
            if key == "E_LT":
                exp_loss[exp_loss >= demolition_threshold*self.replCost] = self.replCost
            E_interpolation_functions[key] = interp1d(IML, exp_loss, fill_value=exp_loss[-1], bounds_error=False)

        return E_interpolation_functions

    def get_eal(self, spline, hazard, method="Porter"):
        """
        Computation of EAL
        :param spline: interp1d             1D interpolation function for the expected total loss
        :param hazard: dict                 Hazard function
        :param method: str                  Calculation method: Porter -> Method 1, other -> applies numpy
                                            (technically the same)
        :return: float                      EAL as a % of the total replacement cost
        :return cache: dict                 EAL bins, IML range and MAF for Visualization
        """
        try:
            # IML step of hazard function
            iml_hazard = np.array(hazard["Sa(T1)"])
            # Ground shaking Mean annual frequency of exceedance
            probs = np.array(hazard["MAFE"])
        except:
            iml_hazard = hazard[1][int(round(self.period * 10))]
            probs = hazard[2][int(round(self.period * 10))]

        # Calling the Cost object
        c = Cost(self.n_stories)
        if method == "Porter":
            # Loss as the ratio of the replacement cost
            mdf = spline(iml_hazard) / self.replCost
            #  Computing the EAL ratio in %
            eal, cache = c.compute_eal(iml_hazard, probs, mdf, rc=1, method=method)

        else:
            # IML
            interpolation = interp1d(iml_hazard, probs)
            iml = np.arange(iml_hazard[0], iml_hazard[-1], 0.01)
            l = interpolation(iml)
            # Loss as the ratio of the replacement cost
            mdf = spline(iml) / self.replCost
            #  Computing the EAL ratio in %
            eal, cache = c.compute_eal(iml, probs, mdf, rc=1, method=method)
        return eal, cache


if __name__ == "__main__":

    # Inputs
    collapse = {"theta": 1.49, "beta": 0.33}
    demolition = {"median": 0.015, "cov": 0.30}
    # Modelling uncertainties
    betas = np.array([0.15, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.6])
    # Total replacement cost of the building
    replCost = 3929937.
    # Set up a main directory
    directory = Path.cwd()
    # Loss object Initialization
    # SLFs are already normalized with respect to the building replacement cost
    filenameX = directory / "client" / "ancona_x.pickle"
    filenameY = directory / "client" / "ancona_y.pickle"
    period = 0.5
    slfFileName = directory / "client" / "slfs.pickle"
    hazardFileName = directory / "client" / "ancona_hazard.csv"
    rsFileName = directory / "client" / "RS.pickle"

    l = Loss(calculate_pga_values=False, nrhaFileName=[filenameX, filenameY], rsFileName=rsFileName,
             hazardFileName=hazardFileName, period=period, slfFileName=slfFileName,
             include_demolition=True, non_directional_factor=1.2, collapse=collapse, demolition=demolition,
             replCost=replCost, betas=betas, performSimulations=False, num_realization=1000)
    # Get start time
    start_time = l.get_init_time()

    # Read the inputs of the framework
    inputs = l.read_input()

    # Calculate losses
    loss = l.calc_losses(inputs["NRHA"], inputs["residuals"])
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

    '''
    # import pickle
    with open("cache.pkl", "wb") as f:
        pickle.dump(cache, f, pickle.HIGHEST_PROTOCOL)

    with open("cache_eal.pkl", "wb") as f:
        pickle.dump(cache_eal, f, pickle.HIGHEST_PROTOCOL)

    with open("loss.pkl", "wb") as f:
        pickle.dump(loss, f, pickle.HIGHEST_PROTOCOL)
    '''








