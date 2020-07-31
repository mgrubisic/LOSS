import pandas as pd
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
import re
from tools.fitting import Fitting
from tools.slf import SLF


class Cost:
    def __init__(self, slf_filename):
        """ 
        :param slf_filename: str                    SLF file name
        """
        self.idr = np.linspace(0, 0.20, 1001)[1:]
        self.acc = np.linspace(0, 20, 1001)[1:]
        self.slf_filename = slf_filename

    def calc_repair_cost(self, edp, max_cost, alfa, beta, gama, delta, epsilon):
        """
        Calculation of repair costs according to the regression by Papadopoulos et al. 2019
        :param edp: list                        EDP range
        :param max_cost: float                  Maximum cost, that is the total cost of the inventory
        :param alfa: float                      Parameter of a regression expression
        :param beta: float                      Parameter of a regression expression
        :param gama: float                      Parameter of a regression expression
        :param delta: float                     Parameter of a regression expression
        :param epsilon: float                   Parameter of a regression expression
        :return: float                          Repair cost value
        """
        # TODO, add support for alternative cost estimation functions, e.g. Weibull
        cost = max_cost * (epsilon * (edp ** alfa) / (beta ** alfa + edp ** alfa) + (1 - epsilon) * (edp ** gama) /
                           (delta ** gama + edp ** gama))
        return cost

    def calc_losses(self, ida_outputs, iml_range, nstories, edp_for_collapse="ISDR", edp_for_demolition="RISDR"):
        """
        Calculates expected losses based on provided storey-loss functions
        :param ida_outputs: dict                    IDA outputs
        :param iml_range: list                      IML range
        :param nstories: int                        Number of stories
        :param edp_for_collapse: str                EDP to process for collapse loss estimation, e.g. ISDR, IDR, PSD
        :param edp_for_demolition: str              EDP to process for demolition loss estimation, e.g. RISDR, RIDR
        :return: dict                               Calculated losses
        """
        """
        Generate DataFrame to store loss results
        Headers:                C = Collapse, D = Demolition
        Headers/per story:      R = Residual, ISDR = inter-story drift ratio, SD - structural damage
                                NSD = non-structural damage, PFA = peak floor acceleration, TOTAL = Total
        """
        df_headers = ['C', 'D']
        df_headers_storey = ['_R_ISDR_SD', '_R_ISDR_NSD', '_R_ISDR_TOTAL', '_R_PFA', '_R_TOTAL']
        for story in np.arange(1, nstories + 1, 1):
            for idx in range(len(df_headers_storey)):
                df_headers.append('%s%s' % (story, df_headers_storey[idx]))
        df_headers += ['R_ISDR_SD_TOTAL', 'R_ISDR_NSD_TOTAL', 'R_ISDR_TOTAL_TOTAL', 'R_PFA_TOTAL', 'R_TOTAL_TOTAL']
        loss_results = pd.DataFrame(columns=df_headers, index=['%.2f' % i for i in iml_range])

        # Collapse losses
        f = Fitting()
        frag_calc = f.calc_collapse_fragility(ida_outputs, edp_to_process=edp_for_collapse)
        theta_col = frag_calc['theta']
        beta_col = frag_calc['beta']
        p_collapse = stats.norm.cdf(np.log(iml_range / theta_col) / beta_col, loc=0, scale=1)
        loss_results['C'] = p_collapse

        # Demolition losses given no collapse - P(D|NC,IM)
        frag_calc = f.calc_demolition_fragility(ida_outputs, iml_range, edp_to_process=edp_for_demolition)
        theta_dem = frag_calc['theta']
        beta_dem = frag_calc['beta']
        p_demol = stats.norm.cdf(np.log(iml_range / theta_dem) / beta_dem, loc=0, scale=1)
        loss_results['D'] = p_demol

        # Getting the SLFs
        slf = SLF(self.slf_filename)
        slf_functions = slf.provided_slf()

        # Repair losses
        # TODO, add support for SLFs varying along the height
        storey_loss_ratio_weights = [1 / nstories for i in range(nstories)]
        for iml in iml_range:
            iml_test = iml
            # Initiate count of repair losses per component group
            r_isdr_sd_total = 0
            r_isdr_nsd_total = 0
            r_isdr_total_total = 0
            r_pfa_total = 0
            r_total_total = 0
            for story in np.arange(1, nstories + 1, 1):
                # Drift-sensitive losses
                edp_to_process = 'ISDR'
                frag_calc = f.calc_p_edp_given_im(ida_outputs, story, edp_to_process, iml, self.idr)
                edp_theta = frag_calc['theta']
                edp_beta = frag_calc['beta']
                p_edp = stats.norm.pdf(np.log(self.idr / edp_theta) / edp_beta, loc=0, scale=1)
                p_edp = p_edp / sum(p_edp)

                # Structural drift-sensitive components
                storey_loss_ratio_idr_sd = sum(slf_functions["E_S_IDR"](self.idr) * p_edp) * \
                                           storey_loss_ratio_weights[story - 1]
                # Non-structural drift-sensitive components
                storey_loss_ratio_idr_nsd = sum(slf_functions["E_NS_IDR"](self.idr) * p_edp) * \
                                            storey_loss_ratio_weights[story - 1]

                # Acceleration-sensitive losses
                edp_to_process = 'PFA'
                storey_loss_ratio_acc = 0
                for floor in [story - 1, story]:
                    frag_calc = f.calc_p_edp_given_im(ida_outputs, floor, edp_to_process, iml, self.acc)
                    edp_theta = frag_calc['theta']
                    edp_beta = frag_calc['beta']
                    p_edp = stats.norm.pdf(np.log(self.acc / edp_theta) / edp_beta, loc=0, scale=1)
                    p_edp = p_edp / sum(p_edp)
                    storey_loss_ratio_acc_partial = sum(slf_functions["E_NS_PFA"](self.acc) * p_edp)
                    storey_loss_ratio_acc_partial /= 2
                    storey_loss_ratio_acc += storey_loss_ratio_acc_partial
                storey_loss_ratio_acc *= storey_loss_ratio_weights[story - 1]

                # Record current repair story losses per component group
                r_isdr_sd = storey_loss_ratio_idr_sd
                r_isdr_nsd = storey_loss_ratio_idr_nsd
                r_isdr_total = r_isdr_sd + r_isdr_nsd
                r_pfa = storey_loss_ratio_acc
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
        loss_results['E_NC_D'] = loss_results['D'] * (1 - loss_results['C'])
        # Collapse
        loss_results['E_C'] = loss_results['C']
        # Total losses, i.e. non-collapse, non-demolition + non-collapse, demolition + collapse
        loss_results['E_LT'] = loss_results['E_NC_ND'] + 1 * loss_results['E_NC_D'] + 1 * loss_results['E_C']

        return loss_results
