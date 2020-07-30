import pickle
import pandas as pd
import numpy as np
from pathlib import Path


class SaT1:
    def __init__(self, rs_path, periods_path, ida_path):
        """
        :param rs_path: str                     Path of the response spectra
        :param periods_path: str                Path of the file containing info on fundamental periods for each case
        :param ida_path: str                    Path of the IDA outputs
        """
        # Get response spectra and periods and IDA outputs
        self.response_spectra = pd.read_pickle(rs_path)
        self.periods = pd.read_csv(periods_path, index_col=None)
        pickle_file = open(ida_path, 'rb')
        self.ida = pickle.load(pickle_file)
        pickle_file.close()

        # Get fundamental period of the case
        self.T1 = float(self.periods.iloc[0])

    def calc_SaT1_mean(self):
        """
        This function gets the mean SaT1 of the building under study for a return period of 475 years (i.e. ULS)
        :return: float                          Spectral Acceleration at T1
        """
        # Get SaT1
        SaT1 = float(self.response_spectra[self.response_spectra['T1'].round(2) == round(self.T1, 2)].loc[:,
                     self.response_spectra.columns != 'T1'].mean(axis=1))
        return SaT1

    def calc_ida_PGA(self):
        """
        This function calculates the PGA associated with each IML of each record
        and stores it in the IDA results pickle (at floor 0 in 'summary_results')
        :return: dict                           Modified IDA outputs with the inclusion of PGA values
        """
        # Copy the ida outputs
        results_mod = self.ida.copy()
        for record in self.ida['summary_results'].keys():
            # Get scaled PGA value
            PGA_scaled = float(self.response_spectra[self.response_spectra['T1'] == 0.00][record])
            # Get SaT1
            SaT1 = float(self.response_spectra[self.response_spectra['T1'].round(2) == round(self.T1, 2)][record])
            # For each intensity measure level
            for iml in sorted(self.ida['summary_results'][record].keys()):
                IDAfact = round(float(iml) / SaT1, 5)
                PGA_scaled_IDA = IDAfact * PGA_scaled
                results_mod['summary_results'][record][iml]['maxFA'][0] = PGA_scaled_IDA
        return results_mod

    def calc_gamma(self, return_period):
        """
        This function calculates the SaT1(RP=475y) modifier for a given return period
        :param return_period: float             Return period in years
        :return: float                          SaT1(RP=475y) modifier for a return_period
        """
        prob_50 = np.exp(-50 / return_period)*(np.exp(50 / return_period) - 1)
        gamma = (prob_50 / 0.1)**(-1 / 3)
        return gamma


if __name__ == "__main__":

    directory = Path.cwd()
    rs_path = directory.parents[0] / "client" / "RS.pickle"
    periods_path = directory.parents[0] / "client" / "Periods.tcl"
    file = directory.parents[0] / "client" / "ida_case1.pickle"

    run = SaT1(rs_path, periods_path, file)
    SaT1 = run.calc_SaT1_mean()
