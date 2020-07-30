"""
user defines storey-loss function parameters
"""
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class SLF:
    def __init__(self, fileName):
        """
        initialize storey loss function definition
        :param fileName: dict                       SLF data file
        """
        self.fileName = fileName

    def provided_slf(self):
        """
        provided as input SLF function
        :return: DataFrame                          Storey loss functions in terms of ELR and EDP
        """
        # TODO, add support for varying functions over the stories
        filename = self.fileName
        df = pd.read_excel(io=filename)
        pfa_ns_range = df['PFA']
        y_ns_pfa_range = df['E_NS_PFA']
        psd_s_range = df['IDR_S']
        y_s_psd_range = df['E_S_IDR']
        psd_ns_range = df['IDR_NS']
        y_ns_psd_range = df['E_NS_IDR']
        y_psd_range = y_s_psd_range + y_ns_psd_range

        # Normalization of storey loss functions
        max_slf = max(y_s_psd_range) + max(y_ns_psd_range) + max(y_ns_pfa_range)
        max_s_psd = max(y_s_psd_range) / max_slf
        max_ns_psd = max(y_ns_psd_range) / max_slf
        max_ns_pfa = max(y_ns_pfa_range) / max_slf

        # Normalized functions
        df["E_NS_PFA"] = df["E_NS_PFA"] * max_ns_pfa / max(y_ns_pfa_range)
        df["E_NS_IDR"] = df["E_NS_IDR"] * max_ns_psd / max(y_ns_psd_range)
        df["E_S_IDR"] = df["E_S_IDR"] * max_s_psd / max(y_s_psd_range)

        return df
