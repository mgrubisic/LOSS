"""
Visualization tool
"""
import matplotlib.pyplot as plt
import subprocess
import os
from pathlib import Path
import seaborn as sns
import numpy as np
from scipy.interpolate import interp1d


class Visualize:

    def __init__(self, figureDirectory=None):
        self.directory = Path(os.getcwd())
        self.color_grid = ['#840d81', '#6c4ba6', '#407bc1', '#18b5d8', '#01e9f5',
                           '#cef19d', '#a6dba7', '#77bd98', '#398684', '#094869']
        self.tick_fontsize = 8
        self.axis_label_fontsize = 8
        self.legend_fontsize = 8
        if figureDirectory is None:
            self.figureDirectory = self.directory / "Data Visualization"
        else:
            self.figureDirectory = figureDirectory

    @staticmethod
    def createFolder(directory):
        """
        Creates a figure if it does not exist
        :param directory: str                       Directory to create
        """
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)

    @staticmethod
    def plot_as_emf(figure, **kwargs):
        """
        Saves figure as .emf
        :param figure: fig handle
        :param kwargs: filepath: str                File name, e.g. '*\filename'
        :return: None
        """
        inkscape_path = kwargs.get('inkscape', "C://Program Files//Inkscape//inkscape.exe")
        filepath = kwargs.get('filename', None)
        if filepath is not None:
            path, filename = os.path.split(filepath)
            filename, extension = os.path.splitext(filename)
            svg_filepath = os.path.join(path, filename + '.svg')
            emf_filepath = os.path.join(path, filename + '.emf')
            figure.savefig(svg_filepath, bbox_inches='tight', format='svg')
            subprocess.call([inkscape_path, svg_filepath, '--export-emf', emf_filepath])
            os.remove(svg_filepath)

    @staticmethod
    def plot_as_png(figure, **kwargs):
        """
        Saves figure as .png
        :param figure: fig handle
        :param kwargs: filepath: str                File name, e.g. '*\filename'
        :return: None
        """
        inkscape_path = kwargs.get('inkscape', "C://Program Files//Inkscape//inkscape.exe")
        filepath = kwargs.get('filename', None)
        if filepath is not None:
            path, filename = os.path.split(filepath)
            filename, extension = os.path.splitext(filename)
            svg_filepath = os.path.join(path, filename + '.svg')
            png_filepath = os.path.join(path, filename + '.png')
            figure.savefig(svg_filepath, bbox_inches='tight', format='svg')
            subprocess.call([inkscape_path, svg_filepath, '--export-png', png_filepath])
            os.remove(svg_filepath)

    def plot_eal(self, cache, losses, sflag=False, pflag=False):
        """
        EAL plots
        :param cache: array                     EAL cache
        :param losses: dict                     Loss ratios
        :param sflag: bool                      Storing as emf flag
        :param pflag: bool                      Plotting flag
        :return: None
        """
        # Reading from cache
        """
        :param eal_bins: array                  EAL steps/bins
        :param iml: array                       IM levels in g
        :param probs: array                     MAF of Hazard
        """
        eal_bins = cache["eal_bins"]
        iml = cache["iml"]
        probs = cache["probs"]

        # EAL contributions with respect to IML
        dIM = np.diff(iml)
        delIM = iml[0:-1] + dIM / 2
        fig1, ax = plt.subplots(figsize=(4, 3), dpi=100)
        plt.bar(delIM, eal_bins / max(eal_bins), width=dIM, color=sns.color_palette("GnBu", len(eal_bins)),
                label="EAL contributions")
        plt.plot(iml, probs/max(probs), color="r", label="Hazard")
        plt.xlabel("IM [g]")
        plt.ylabel("Normalized MAF")
        plt.xlim(iml[0], 1)
        plt.ylim(0, 1)
        plt.grid(True, which="major", axis="y", ls="--", lw=1.0)
        plt.legend(frameon=False, loc="upper right", fontsize=self.legend_fontsize)
        if not pflag:
            plt.close()
        if sflag:
            name = self.figureDirectory / "EAL_Contributions"
            self.plot_as_emf(fig1, filename=name)

        # EAL disaggregation by NS repair, S repair, Collapse and Demolition expected loss
        tags = ["E_NC_ND_NS", "E_NC_ND_S", "E_C", "E_NC_D", "E_LT"]
        # Creating labels for plotting
        labels = []
        for i in tags:
            if i == "E_LT":
                label = r'$L_{T}$ (Total Loss)'
            elif i == "E_NC_D":
                label = r'$L_{NC \cap D}$ (Non Collapse-Demolition)'
            elif i == "E_C":
                label = r'$L_{C}$ (Collapse Loss)'
            elif i == "E_NC_ND_S":
                label = r'$L_{NC, S \cap R}$ (Non Collapse Structural-Repair)'
            else:
                label = r'$L_{NC, NS \cap R}$ (Non Collapse Non-Structural-Repair)'
            labels.append(label)

        # Get the IML range for plotting
        IML = np.insert(np.array([float(i) for i in losses.index]), 0, 0)
        EALs = []
        mdf = np.zeros((len(tags), len(iml)))
        for j in range(mdf.shape[0]):
            # Get the loss ratios
            loss = np.array(losses[tags[j]])
            # Set an interpolation function
            spline = interp1d(IML, np.insert(loss, 0, 0))
            # Loss ratios
            mdf[j, :] = spline(iml)
            # Initialize EAL bins
            dEAL = []
            for i in range(len(probs) - 1):
                dIM = iml[i + 1] - iml[i]
                delIM = iml[i] + dIM / 2
                dLdIM = np.log(probs[i + 1] / probs[i]) / dIM
                dMDF = mdf[j, i + 1] - mdf[j, i]
                dEAL.append(mdf[j, i] * probs[i] * (1 - np.exp(dLdIM * dIM)) - dMDF / dIM * probs[i] *
                            (np.exp(dLdIM * dIM) * (dIM - 1 / dLdIM) + 1 / dLdIM))
            if tags[j] == "E_LT" or tags[j] == "E_C":
                # Add only once
                add = probs[-1]
            else:
                add = 0
            # EAL contributions from each category
            EALs.append((sum(dEAL) + add) * 100)

        # Plotting
        EALs = np.flip(EALs, axis=0)
        EALs = np.delete(EALs, obj=0, axis=0)
        fig2, ax = plt.subplots(figsize=(4, 3), dpi=100)
        cnt = 0
        for i in range(len(EALs)):
            if i == 0:
                bottom = 0
            else:
                bottom = sum(EALs[:i])
            plt.bar(0.5, EALs[i], width=0.15, bottom=bottom, edgecolor='none', zorder=1000,
                    label=labels[i], color=self.color_grid[cnt])
            cnt += 2
        plt.ylabel("Expected Annual Loss Ratio [%]")
        plt.xlim(0.3, 0.7)
        plt.ylim(0, max(EALs) + 0.2)
        plt.xticks([])
        plt.yticks(np.linspace(0, np.round(max(EALs) + 0.2, 1), 4))
        plt.grid(True, which="major", axis="y", ls="--", lw=1.0)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if not pflag:
            plt.close()
        if sflag:
            name = self.figureDirectory / "EAL"
            self.plot_as_emf(fig2, filename=name)

    def plot_loss_curves(self, losses, sflag=False, pflag=False):
        """
        Plots loss curves / vulnerability functions disaggregated by component groups, collapse and demolition
        :param losses: dict                     Loss ratios
        :param sflag: bool                      Storing as emf flag
        :param pflag: bool                      Plotting flag
        :return: None
        """
        # Get the IML range for plotting
        IML = np.insert(np.array([float(i) for i in losses.index]), 0, 0)

        tags = ["E_NC_ND_NS", "E_NC_ND_S", "E_C", "E_NC_D", "E_LT"]
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        plt.ylabel('Normalized losses')
        plt.xlabel(r'$IM = S_a(T_1,5$%$)\ [g]$', fontsize=self.axis_label_fontsize, labelpad=10)
        plt.ylim(0, 1)
        plt.xlim(0, 3.5)
        plt.grid(True, which="major", axis="both", ls="--", lw=1.0)
        cnt = 0
        for i in tags:
            loss = np.insert(np.array(losses[i]), 0, 0)
            if i == "E_LT":
                label = r'$L_{T}$ (Total Loss)'
            elif i == "E_NC_D":
                label = r'$L_{NC \cap D}$ (Non Collapse-Demolition)'
            elif i == "E_C":
                label = r'$L_{C}$ (Collapse Loss)'
            elif i == "E_NC_ND_S":
                label = r'$L_{NC, S \cap R}$ (Non Collapse Structural-Repair)'
            else:
                label = r'$L_{NC, NS \cap R}$ (Non Collapse Non-Structural-Repair)'

            plt.plot(IML, loss,
                     label=label,
                     lw=1.0, linestyle='-', color=self.color_grid[cnt],
                     marker="s", markersize=4, markevery=4)
            cnt += 2
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if not pflag:
            plt.close()
        if sflag:
            name = self.figureDirectory / "Loss_curves"
            self.plot_as_emf(fig, filename=name)

    def plot_vulnerability(self, cache, sflag=False, pflag=False):
        """
        Plots the vulnerability curve
        :param cache: array                     EAL cache
        :param sflag: bool                      Storing as emf flag
        :param pflag: bool                      Plotting flag
        :return: None
        """
        # Reading from cache
        """
        :param elr: array                       Expected loss ratios, ELR
        :param probs: array                     MAF of Hazard
        """
        probs = cache["probs"]
        elr = cache["mdf"]
        # Return periods
        rp = 1 / probs

        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        plt.plot(rp, elr, color=self.color_grid[0], marker="o")
        ax.set_xscale("log")
        plt.grid(True, which="both", axis="both", ls="--", lw=1.0)
        plt.xlabel("Return Period [years]")
        plt.ylabel("Expected Loss Ratio")
        plt.ylim(0, 1.0)
        plt.xlim(10, 10 ** 4)
        if not pflag:
            plt.close()
        if sflag:
            name = self.figureDirectory / "vulnerability"
            self.plot_as_emf(fig, filename=name)

    def area_plots(self, cache, losses, sflag=False, pflag=False):
        """
        Makes an area plot of contributions to the Expected loss ratio, ELR
        :param cache: array                     EAL cache
        :param losses: dict                     Losses
        :param sflag: bool                      Storing as emf flag
        :param pflag: bool                      Plotting flag
        :return: None
        """
        # Reading from cache
        """
        :param iml: array                       IML range
        :param probs: array                     MAF of Hazard
        """
        probs = cache["probs"]
        iml = cache["iml"]

        # Get the IML range for plotting
        IML = np.insert(np.array([float(i) for i in losses.index]), 0, 0)
        tags = ["E_NC_ND_NS", "E_NC_ND_S", "E_C", "E_NC_D"]
        # Interpolation functions
        spline = interp1d(IML, np.insert(np.array(losses['E_LT']), 0, 0))
        # Total loss ratio
        loss_t = spline(iml)
        # Return periods
        rp = 1 / probs
        # Loss contributions
        loss_contr = np.zeros((4, loss_t.shape[0]))
        for i in range(loss_contr.shape[0]):
            spline = interp1d(IML, np.insert(np.array(losses[tags[i]]), 0, 0))
            loss_contr[i, :] = spline(iml) / loss_t
        loss_contr = np.flip(loss_contr, axis=0)
        # Plotting
        fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
        labels = [r'$L_{NC, NS \cap R}$ (Non Collapse Non-Structural-Repair)',
                  r'$L_{NC, S \cap R}$ (Non Collapse Structural-Repair)',
                  r'$L_{C}$ (Collapse Loss)',
                  r'$L_{NC \cap D}$ (Non Collapse-Demolition)']
        ax = plt.gca()
        ax.stackplot(rp, loss_contr, labels=labels, colors=self.color_grid)
        ax.set_xscale("log")
        ax.set_ylim([0, 1])
        ax.set_xlim([rp[0], rp[-1]])
        plt.grid(True, which="both", axis="both", ls="--", lw=1.0)
        ax.set_xlabel("Return Period [years]")
        ax.set_ylabel("Contribution to ELR")

        # Lighten borders
        plt.gca().spines["top"].set_alpha(0)
        plt.gca().spines["bottom"].set_alpha(.3)
        plt.gca().spines["right"].set_alpha(0)
        plt.gca().spines["left"].set_alpha(.3)

        # Decorations
        ax.legend(loc='center left', fontsize=8, ncol=1, bbox_to_anchor=(1, 0.5))
        rp_to_plot = [100, 475, 2475]
        for i in rp_to_plot:
            ax.plot([i, i], [0, 1], color='k')
            ax.annotate(f"{i} years", xy=(i, 0.95), rotation=-90, va='top', ha='left', fontsize=self.legend_fontsize,
                        color='k')
        if not pflag:
            plt.close()
        if sflag:
            name = self.figureDirectory / "area"
            self.plot_as_emf(fig, filename=name)
