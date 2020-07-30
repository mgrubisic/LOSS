import os
from pathlib import Path
import subprocess
import numpy as np
import pickle
import re
import scipy
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from tools.cost import Cost
from tools.fitting import Fitting
from tools.sat1 import SaT1


class Loss:
    def __init__(self, directory=None, idaFileName=None, rsFileName=None, periodsFileName=None, slfFileName=None,
                 iml_max=5.0, iml_step=0.05, calculate_pga_values=False):
        """
        Initialize Loss estimation framework based on story-loss functions
        :param directory: str                           Main working directory
        :param idaFileName: str                         Incremental dynamic analysis (IDA) filename
        :param rsFileName: str                          Response spectra (RS) file name
        :param periodsFileName: str                     File name containing 1st mode periods of the buildings
        :param slfFileName: str                         SLF file name
        :param iml_max: float                           Max IML
        :param iml_step: float                          IML step
        :param calculate_pga_values: bool               Whether to calculate the PGA values (usually True if none
                                                        provided)
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

        # Get the intensity measure level (IML) range
        self.iml_range = np.arange(iml_step, iml_max + iml_step, iml_step)

        # Calculation of PGA values
        self.calculate_pga_values = calculate_pga_values

        # Initialize number of stories
        self.nstories = None

    def read_input(self):
        """
        Reads the inputs for the loss assessment
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
            ida_file = open(clientDirectory / self.idaFileName, "rb")
            ida_outputs = pickle.load(ida_file)
            ida_file.close()

        # Reading of the RS file
        if self.rsFileName is None:
            for file in os.listdir(clientDirectory):
                if file.startswith("RS") or file.startswith("rs"):
                    rs = pd.read_pickle(clientDirectory / file)
                    self.rsFileName = clientDirectory / file
        else:
            rs = pd.read_pickle(self.rsFileName)

        # Get the fundamental (1st mode) periods of the buildings under consideration
        if self.periodsFileName is None:
            for file in os.listdir(clientDirectory):
                if file.startswith("period") or file.startswith("Period"):
                    periods = pd.read_csv(clientDirectory / file, index_col=None)
                    self.periodsFileName = clientDirectory / file
        else:
            periods = pd.read_csv(self.periodsFileName, index_col=None)

        # Get number of stories of the building
        for i in ida_outputs["summary_results"]:
            for j in ida_outputs["summary_results"][i].keys():
                for k in ida_outputs["summary_results"][i][j].keys():
                    for key in ida_outputs["summary_results"][i][j].get(k):
                        self.nstories = key
                    break
                break
            break

        # TODO, add batch mode, where multiple buildings will be analyzed
        # Check whether peak ground acceleration (PGA) values are provided, if not, calculate them
        if self.calculate_pga_values:
            ida_outputs = self.calc_PGA()

        return {"IDA": ida_outputs, "RS": rs, "Periods": periods, "Nstories": self.nstories}

    def calc_PGA(self):
        """
        Calculate the PGA values
        :return: dict                               IDA outputs
        """
        sat1 = SaT1(self.rsFileName, self.periodsFileName, self.idaFileName)
        outputs = sat1.calc_ida_PGA()
        return outputs

    def calc_loses(self, ida_outputs):
        """
        Calculates losses based on SLFs
        :param ida_outputs:                         IDA outputs
        :return: dict                               Computed, disaggregated and total losses
        """
        # Get client directory
        clientDirectory = self.directory / "client"

        if self.slfFileName is None:
            for file in os.listdir(clientDirectory):
                if file.startswith("slf"):
                    cost = Cost(slf_filename=clientDirectory / file)
        else:
            cost = Cost(slf_filename=clientDirectory / self.slfFileName)
        losses = cost.calc_losses(ida_outputs, self.iml_range, self.nstories)
        return losses


if __name__ == "__main__":

    directory = Path.cwd()
    l = Loss(calculate_pga_values=False)
    inputs = l.read_input()
    a = l.calc_loses(inputs["IDA"])
    print(a)
