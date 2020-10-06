import pandas as pd
import numpy as np
from scipy import stats
from scipy.integrate import trapz
from tools.fitting import Fitting
from tools.slf import SLF


class Cost:
    def __init__(self, slf_filename=None, include_demolition=False, nonDirFactor=1.0):
        """ 
        :param slf_filename: str                    SLF file name
        :param include_demolition: bool             Whether to calculate loss contribution from demolition
        :param nonDirFactor: float                  Non-directional conversion factor
        """
        # TODO, make more generic; actual story-loss functions might not be within the range, so it might give an error
        self.idr = np.linspace(0, 20.0, 1001)[1:]
        self.acc = np.linspace(0, 10, 1001)[1:]
        self.slf_filename = slf_filename
        self.include_demolition = include_demolition
        self.nonDirFactor = nonDirFactor
        self.nstories = None
        
    def get_non_dir_ida_results(self, ida_outputs):
        """
        Transforms IDA demands by a non-dimensional factor
        :param ida_outputs:                         IDA outputs
        :returns: dict                              Transformed IDA outputs
        """
        nonDirFactor = self.nonDirFactor
        
        # Get max demand and multiply by nonDirFactor
        ida = {}
        direction_keys = list(ida_outputs.keys())
        # ASSUMPTION: IML levels are the same, TODO, too rigid of a condition for a code to work for flexible inputs
        for key in ida_outputs[direction_keys[0]].keys():
            ida[key] = {}
            for gm in ida_outputs[direction_keys[0]][key].keys():
                ida[key][gm] = {}
                if key == "IDA":
                    for e in ida_outputs[direction_keys[0]][key][gm].keys():
                        if e == "ISDR" or e == "PFA":
                            ida[key][gm][e] = np.amax(np.stack((ida_outputs[direction_keys[0]][key][gm][e],
                                                                ida_outputs[direction_keys[1]][key][gm][e])),
                                                      axis=0) * nonDirFactor
                        else:
                            ida[key][gm][e] = ida_outputs[direction_keys[0]][key][gm][e]
                else:
                    for i in ida_outputs[direction_keys[0]][key][gm].keys():
                        ida[key][gm][i] = {}
                        for e in ida_outputs[direction_keys[0]][key][gm][i].keys():
                            if e == "maxISDR" or e == "maxFA":
                                ida[key][gm][i][e] = {}
                                for st in ida_outputs[direction_keys[0]][key][gm][i][e].keys():
                                    ida[key][gm][i][e][st] = max(ida_outputs[direction_keys[0]][key][gm][i][e][st],
                                                                 ida_outputs[direction_keys[1]][key][gm][i][e][st])\
                                                             * nonDirFactor
                            else:
                                ida[key][gm][i][e] = {}
                                for st in ida_outputs[direction_keys[0]][key][gm][i][e].keys():
                                    ida[key][gm][i][e][st] = ida_outputs[direction_keys[0]][key][gm][i][e][st]
        return ida

    def get_drift_sensitive_losses(self, ida_outputs, story, iml, interpolation, directionality=False,
                                   edp_to_process="ISDR", direction=None):
        """
        Calculate drift sensitive story loss ratios
        :param ida_outputs: dict                    IDA outputs
        :param story: int                           Story level
        :param iml: float                           IM level
        :param interpolation: dict                  Interpolation functions
        :param directionality: bool                 Directionality mask
        :param edp_to_process: str                  EDP to process
        :param direction: str                       Direction for accessing interpolation functions
        :return: float                              Story loss ratios
        """

        # Call the Fitting object
        f = Fitting()

        # Check directionality of component groups
        if directionality:
            direction_tag = "Directional"

            # Probabilities of exceedance
            frag_calc = f.calc_p_edp_given_im(ida_outputs, story, edp_to_process, iml, self.idr)
            edp_theta = frag_calc['theta']
            edp_beta = frag_calc['beta']
            p_edp = stats.norm.pdf(np.log(self.idr / edp_theta) / edp_beta, loc=0, scale=1)
            p_edp = p_edp / sum(p_edp)

            # Apply interpolation
            for edp in interpolation[direction_tag].keys():
                if edp == "IDR_S":
                    cost_sd = sum(interpolation[direction_tag][edp][direction][story](self.idr / 100)
                                                  * p_edp)
                elif edp == "IDR_NS":
                    cost_nsd = sum(interpolation[direction_tag][edp][direction][story](self.idr / 100)
                                                   * p_edp)
            return cost_sd, cost_nsd

        else:
            direction_tag = "Non-directional"

            # Get probabilities of exceedance
            frag_calc = f.calc_p_edp_given_im(ida_outputs, story, edp_to_process, iml, self.idr)
            edp_theta = frag_calc['theta']
            edp_beta = frag_calc['beta']
            p_edp = stats.norm.pdf(np.log(self.idr / edp_theta) / edp_beta, loc=0, scale=1)
            p_edp = p_edp / sum(p_edp)

            # Apply interpolation
            for edp in interpolation[direction_tag].keys():
                if edp == "IDR_S":
                    cost_sd = sum(interpolation[direction_tag][edp][story](self.idr / 100) * p_edp)
                elif edp == "IDR_NS":
                    cost_nsd = sum(interpolation[direction_tag][edp][story](self.idr / 100) * p_edp)

            # Storing the already modified IDA results to be used for the PFA-sensitive non-directional components
            costs = [cost_sd, cost_nsd]
            
            return costs

    def get_acceleration_sensitive_losses(self, ida_outputs, floor, iml, interpolation, edp_to_process="PFA"):
        """
        Get acceleration sensitive story loss ratios
        :param ida_outputs: dict                IDA outputs already modified to account for non-directionality
        :param floor: int                       Floor level
        :param iml: float                       IM level
        :param interpolation: dict              Interpolation functions for PFA-sensitive SLFs
        :param edp_to_process: str              EDP tag, default "PFA"
        :return: float                          Story loss ratio for the IML of interest
        """
        # Call the Fitting object
        f = Fitting()

        # Probabilities of exceedance
        frag_calc = f.calc_p_edp_given_im(ida_outputs, floor, edp_to_process, iml, self.acc)
        edp_theta = frag_calc['theta']
        edp_beta = frag_calc['beta']
        p_edp = stats.norm.pdf(np.log(self.acc / edp_theta) / edp_beta, loc=0, scale=1)
        p_edp = p_edp / sum(p_edp)
        story_loss_ratio_acc_partial = sum(interpolation[floor](self.acc) * p_edp)
        if floor != 0 and floor != self.nstories:
            story_loss_ratio_acc_partial /= 2
        cost = story_loss_ratio_acc_partial
        return cost

    def calc_losses(self, ida_outputs, iml_range, nstories, edp_for_collapse="ISDR", edp_for_demolition="RISDR", 
                    collapse=None, demolition=None, use_beta_MDL=0.15, replCost=1):
        """
        Calculates expected losses based on provided storey-loss functions
        :param ida_outputs: dict                    IDA outputs
        :param iml_range: list                      IML range
        :param nstories: int                        Number of stories
        :param edp_for_collapse: str                EDP to process for collapse loss estimation, e.g. ISDR, IDR, PSD
        :param edp_for_demolition: str              EDP to process for demolition loss estimation, e.g. RISDR, RIDR
        :param collapse: dict                       Median and dispersion of collapse fragility function, or None if needs to be computed
        :param demolition: dict                     Median and dispersion of dispersion fragility function, or None if default is used
        :param use_beta_MDL: float                  Standard deviation accounting for modelling uncertainty to use
        :param replCost: float                      Replacement cost of the building
        :return: dict                               Calculated losses
        """
        """
        Generate DataFrame to store loss results
        Headers:                C = Collapse, D = Demolition
        Headers/per story:      R = Residual, ISDR = inter-story drift ratio, SD - structural damage
                                NSD = non-structural damage, PFA = peak floor acceleration, TOTAL = Total
        """
        self.nstories = nstories
        
        # Get IDA results for Non-dimensional components
        ida_non_dir = self.get_non_dir_ida_results(ida_outputs)
        
        # Initialize loss dictionary with appropriate headers
        df_headers = ['C', 'D']
        df_headers_storey = ['_R_ISDR_SD', '_R_ISDR_NSD', '_R_ISDR_TOTAL', '_R_PFA', '_R_TOTAL']
        for story in np.arange(1, nstories + 1, 1):
            for idx in range(len(df_headers_storey)):
                df_headers.append('%s%s' % (story, df_headers_storey[idx]))
        df_headers += ['R_ISDR_SD_TOTAL', 'R_ISDR_NSD_TOTAL', 'R_ISDR_TOTAL_TOTAL', 'R_PFA_TOTAL', 'R_TOTAL_TOTAL']
        loss_results = pd.DataFrame(columns=df_headers, index=['%.2f' % i for i in iml_range])
        
        # Call the Fitting object
        f = Fitting()
        # Collapse losses
        if collapse is None:
            # TODO, currently calling only x, to be modified to be based on the average of both x and y demands, or
            #  the max (needs to be checked and updated)
            frag_calc = f.calc_collapse_fragility(ida_outputs["x"], edp_to_process=edp_for_collapse,
                                                  use_beta_MDL=use_beta_MDL)
            theta_col = frag_calc['theta']
            beta_col = frag_calc['beta']
            p_collapse = stats.norm.cdf(np.log(iml_range / theta_col) / beta_col, loc=0, scale=1)
            loss_results['C'] = p_collapse
        else:
            # Dispersion includes modelling uncertainty as well
            theta_col = collapse['theta']
            beta_col = collapse['beta']
            p_collapse = stats.norm.cdf(np.log(iml_range / theta_col) / beta_col, loc=0, scale=1)
            loss_results['C'] = p_collapse
        
        # Demolition losses given no collapse - P(D|NC,IM)
        if demolition is None:
            ls_median = None
            ls_cov = None
        else:
            ls_median = demolition["median"]
            ls_cov = demolition["cov"]
            
        if self.include_demolition:
            # TODO, in the IDA outputs currently, RIDR is not dependent on the direction, so x is being used here
            frag_calc = f.calc_demolition_fragility(ida_outputs["x"], iml_range, edp_to_process=edp_for_demolition,
                                                    ls_median=ls_median, ls_cov=ls_cov)
            theta_dem = frag_calc['theta']
            beta_dem = frag_calc['beta']
            p_demol = stats.norm.cdf(np.log(iml_range / theta_dem) / beta_dem, loc=0, scale=1)
            loss_results['D'] = p_demol
        else:
            loss_results['D'] = 0
        
        # Getting the SLFs
        slf = SLF(self.slf_filename, nstories)
        slf_functions = slf.provided_slf()
        interpolation_functions = slf.get_interpolation_functions(slf_functions)

        # Repair losses
        # Loop for each intensity measure level (IML)
        # TODO, try getting rid of a loop for each iml, vectorize, otherwise very time-consuming
        for iml in iml_range:
            iml_test = iml
            # Initiate count of repair losses per component group
            r_isdr_sd_total = 0
            r_isdr_nsd_total = 0
            r_isdr_total_total = 0
            r_pfa_total = 0
            r_total_total = 0

            # Loop through each story
            for story in np.arange(1, nstories + 1, 1):

                # Initiate count of story loss ratios for each component group
                story_loss_ratio_idr_sd = 0
                story_loss_ratio_idr_nsd = 0
                story_loss_ratio_acc = 0

                # Loop for directional and non-directional components
                for dirType in interpolation_functions.keys():
                    if dirType == "Non-directional":

                        # Drift-sensitive losses
                        temp = self.get_drift_sensitive_losses(ida_non_dir, story, iml, interpolation_functions)
                        
                        story_loss_ratio_idr_sd += temp[0]
                        story_loss_ratio_idr_nsd += temp[1]

                        # Acceleration-sensitive losses
                        for floor in [story - 1, story]:
                            temp = self.get_acceleration_sensitive_losses(ida_non_dir, floor, iml,
                                                                          interpolation_functions["Non-directional"][
                                                                              "PFA_NS"], "PFA")
                            story_loss_ratio_acc += temp
                    else:
                        # Drift-sensitive losses
                        for key in ida_outputs.keys():
                            if key == "x":
                                direction = "dir1"
                            else:
                                direction = "dir2"

                            temp = self.get_drift_sensitive_losses(ida_outputs[key], story, iml,
                                                                   interpolation_functions, directionality=True,
                                                                   direction=direction)
                            
                            story_loss_ratio_idr_sd += temp[0]
                            story_loss_ratio_idr_nsd += temp[1]

                # Record current repair story losses per component group
                r_isdr_sd = story_loss_ratio_idr_sd
                r_isdr_nsd = story_loss_ratio_idr_nsd
                r_isdr_total = r_isdr_sd + r_isdr_nsd
                r_pfa = story_loss_ratio_acc
                r_total = r_isdr_total + r_pfa

                # Update the dictionary of the repair loss outputs
                columns = ['%s%s' % (story, i) for i in df_headers_storey]
                values = [r_isdr_sd, r_isdr_nsd, r_isdr_total, r_pfa, r_total]
                loss_results.loc['%.2f' % iml_test, columns] = values

                # Update the total repair losses per component group
                r_isdr_sd_total += r_isdr_sd
                r_isdr_nsd_total += r_isdr_nsd
                r_isdr_total_total += r_isdr_total
                r_pfa_total += r_pfa
                r_total_total += r_total

            # Create headers for the repair losses per component group type and assign the computed values
            columns = ['R_ISDR_SD_TOTAL', 'R_ISDR_NSD_TOTAL', 'R_ISDR_TOTAL_TOTAL', 'R_PFA_TOTAL',
                       'R_TOTAL_TOTAL']
            values = [r_isdr_sd_total, r_isdr_nsd_total, r_isdr_total_total, r_pfa_total, r_total_total]
            loss_results.loc['%.2f' % iml_test, columns] = values

        """Compute the expected total losses per component group.
        Disaggregation of losses"""
        # Non-collapse, non-demolition, IDR-sensitive structural components
        loss_results['E_NC_ND_ISDR_S'] = loss_results['R_ISDR_SD_TOTAL'] * (1 - loss_results['C']) * (
                1 - loss_results['D'])
        # Non-collapse, non-demolition, IDR-sensitive non-structural components
        loss_results['E_NC_ND_ISDR_NS'] = loss_results['R_ISDR_NSD_TOTAL'] * (1 - loss_results['C']) * (
                1 - loss_results['D'])
        # Non-collapse, non-demolition, IDR-sensitive total
        loss_results['E_NC_ND_ISDR_TOTAL'] = loss_results['R_ISDR_TOTAL_TOTAL'] * (1 - loss_results['C']) * (
                1 - loss_results['D'])
        # Non-collapse, non-demolition, PFA-sensitive total
        loss_results['E_NC_ND_PFA_TOTAL'] = loss_results['R_PFA_TOTAL'] * (1 - loss_results['C']) * (
                1 - loss_results['D'])
        # Non-collapse, non-demolition, structural components
        loss_results['E_NC_ND_S'] = loss_results['E_NC_ND_ISDR_S']
        # Non-collapse, non-demolition, non-structural components
        loss_results['E_NC_ND_NS'] = loss_results['E_NC_ND_ISDR_NS'] + loss_results['E_NC_ND_PFA_TOTAL']
        # Non-collapse, non-demolition
        loss_results['E_NC_ND'] = loss_results['E_NC_ND_S'] + loss_results['E_NC_ND_NS']
        # Non-collapse, demolition
        loss_results['E_NC_D'] = loss_results['D'] * (1 - loss_results['C']) * replCost
        # Collapse
        loss_results['E_C'] = loss_results['C'] * replCost
        # Total losses, i.e. non-collapse, non-demolition + non-collapse, demolition + collapse
        loss_results['E_LT'] = loss_results['E_NC_ND'] + 1 * loss_results['E_NC_D'] + 1 * loss_results['E_C']

        return loss_results

    def compute_eal(self, iml, l, mdf, rc=1.0, method="Porter"):
        """ 
        Computation of EAL
        :param iml: array                   IM levels in [g]
        :param l: array                     Hazard curve (MAFE)
        :param mdf: array                   Loss as a ratio of replacement cost
        :param rc: float                    Replacement cost
        :param method: str                  Calculation method: Porter -> Method 1, other -> applies numpy (technically the same)
        :return eal_ratio: float            EAL as a % of the total replacement cost
        :return cache: dict                 EAL bins, IML range and MAF for Visualization
        """
        if method == "Porter":
            # Initialize contributions of EAL
            eal_bins = np.array([])
            # Initialize midpoint generation on IML
            im_midpoint = np.array([])
            for i in range(len(l) - 1):
                # IML step
                dIM = iml[i+1] - iml[i]
                # Midpoint of consecutive IM levels
                im_midpoint = np.append(im_midpoint, iml[i] + dIM / 2)
                # dMAFEdIM, logarithmic gradient divided by IML step
                dLdIM = np.log(l[i+1] / l[i]) / dIM
                # Loss ratio step
                dMDF = mdf[i+1] - mdf[i]
                # EAL contribution at each level of IML
                eal_bins = np.append(eal_bins, (mdf[i] * l[i] * (1 - np.exp(dLdIM*dIM)) - dMDF/dIM*l[i] *
                                                (np.exp(dLdIM*dIM)*(dIM - 1/dLdIM) + 1/dLdIM)))
            # EAL expressed in the units of total replacement cost
            eal = rc * (sum(eal_bins) + l[-1])
            # EAL ratio in % of the total building cost (rc)
            eal_ratio = eal / rc * 100

            # Cache
            cache = {"eal_bins": eal_bins, "iml": iml, "probs": l, "mdf": mdf}
            
        else:
            # IM level step
            dIM = iml[1] - iml[0]
            # Slopes
            slopes = abs(np.gradient(l, dIM))
            # EAL calculation through the trapezoidal rule
            eal_ratio = trapz(y=mdf*slopes, x=iml) * 100

            cache = None

        return eal_ratio, cache




















