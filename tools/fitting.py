import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d
from scipy import stats


class Fitting:
    def __init__(self):
        pass

    def spline(self, edp, iml, length=2000):
        """
        Perform a spline
        :param edp: list                EDP array
        :param iml: list                IM level array
        :param length: int              Length of the interpolation function
        :return: dict                   Fitted EDP vs IML functions
        """
        i = np.arange(len(edp))
        interp_i = np.linspace(0, i[-1], length * i[-1])
        xi = interp1d(i, edp)(interp_i)
        yi = interp1d(i, iml)(interp_i)
        return {'edp_spline': xi, 'iml_spline': yi}

    def mlefit(self, theta, num_recs, num_collapse, IM):
        """
        Performs a lognormal CDF fit to fragility data points based on maximum likelihood method
        :param theta: float             Medians and standard deviations of the function
        :param num_recs: int            Number of records
        :param num_collapse: int        Number of collapses
        :param IM: list                 Intensity measures
        :return: float
        """
        p = stats.norm.cdf(np.log(IM), loc=np.log(theta[0]), scale=theta[1])
        likelihood = stats.binom.pmf(num_collapse, num_recs, p)
        loglik = -sum(np.log10(likelihood))
        return loglik

    def calc_collapse_fragility(self, ida_results, edp_to_process="ISDR", flat_slope=.1, use_edp_max=10.,
                                use_beta_MDL=.15):
        """
        Calculate a IMLvsPoE fragility for the collapse limit state
        :param ida_results: dict        IDA outputs
        :param edp_to_process: str      Type of EDP to process, e.g. ISDR
        :param flat_slope: float        Flattening slope
        :param use_edp_max: float       EDP max to use
        :param use_beta_MDL: float      Standard deviation accounting for modelling uncertainty to use
        :return: dict                   Fitting function parameters
        """
        # Calculate and plot limit state exceedance points on IDA, edp_max in %, iml_max in g
        edp_max = max([max(ida_results['IDA'][record][edp_to_process]) for record in ida_results['IDA'] if
                       len(ida_results['IDA'][record]) != 0])
        iml_max = max([max(ida_results['IDA'][record]['IM']) for record in ida_results['IDA'] if
                       len(ida_results['IDA'][record]) != 0])
        edp_max_mod = edp_max * 2
        iml_max_mod = iml_max
        exceeds = []
        flats = []
        for record in ida_results['IDA']:
            if len(ida_results['IDA'][record]) != 0:
                # IM levels per record
                iml = np.array(ida_results['IDA'][record]['IM'])
                # EDP values per record
                edp = np.array(ida_results['IDA'][record][edp_to_process])
                
                # Get the order of the IML
                order = iml.argsort()
                edp = edp[order]
                iml = iml[order]
                # Flatline
                edp = np.append(edp, edp_max_mod)
                iml = np.append(iml, iml[-1])
                
                # Create a spline of edp vs iml
                spline = self.spline(edp, iml)
                edp_news = spline['edp_spline']
                iml_news = spline['iml_spline']
                slope_ratio = flat_slope
                slope_init = iml[1] / edp[1]
                slopes = np.diff(iml_news) / np.diff(edp_news)
                flat_idx = np.where(slopes == slopes[(slopes < slope_ratio * slope_init) & (slopes > 0) &
                                                     (slopes != np.inf)][0])[0][0]
                flat_iml = iml_news[flat_idx - 1]
                flat_edp = edp_news[flat_idx - 1]
                flat_edp_lim = use_edp_max
                if flat_edp > flat_edp_lim:
                    flat_edp = flat_edp_lim
                    flat_iml = float(iml_news[np.where(edp_news == edp_news[edp_news > flat_edp_lim][0])[0]])
                exceeds.append(flat_iml)
                flats.append((record, flat_iml))
        iml_min = min(exceeds)
        iml_max = max(exceeds)
        # Fragility calculations with MLE fitting
        iml_news_all = np.linspace(0, iml_max_mod, 100)
        counts = []
        for iml_new in iml_news_all:
            if iml_new < iml_min:
                count = 0
            elif iml_new > iml_max:
                count = len(exceeds)
            else:
                count = sum(np.array(exceeds) < iml_new)
            counts.append(count)
        num_recs = len(exceeds)
        xs = iml_news_all
        ys = counts
        if xs[0] == 0:
            xs, ys = xs[1:], counts[1:]
        theta_hat_mom = np.exp(np.mean(np.log(xs)))
        beta_hat_mom = np.std(np.log(xs))
        x0 = [theta_hat_mom, beta_hat_mom]
        
        xopt_mle = optimize.fmin(func=lambda var: self.mlefit(theta=[var[0], var[1]], num_recs=num_recs,
                                                              num_collapse=np.array(ys[1:]), IM=xs[1:]),
                                 x0=x0, maxiter=3000, maxfun=3000, disp=False)
        theta_mle = xopt_mle[0]
        # Applying BetaTOT from Tzimas et al 2016
        # BetaRTR from IDA "+" BetaMDL
        beta_mle = (xopt_mle[1] ** 2 + use_beta_MDL ** 2) ** 0.5
        return {'theta': theta_mle, 'beta': beta_mle, 'flats': flats}

    def calc_demolition_fragility(self, ida_results, iml_range, edp_to_process="RISDR", ls_median=None, ls_cov=None):
        """
        This function calculates a IMLvsPoE fragility for the demolition limit state
        :param ida_results: dict            IDA outputs
        :param iml_range: list              IML range
        :param edp_to_process: str          Type of EDP to process, e.g. RISDR
        :param ls_median: float             Median EDP
        :param ls_cov: float                EDP covariance
        :return: dict                       Fitting function parameters
        """
        if ls_median is None:
            ls_median = 0.0185
        if ls_cov is None:
            ls_cov = 0.3
            
        iml_max = max([max(ida_results['IDA'][record]['IM']) for record in ida_results['IDA'] if
                       len(ida_results['IDA'][record]) != 0])

        # Update IML range if the maximum selected is lower than the actual values obtained
        if max(iml_range) <= iml_max:
            step = iml_range[1] - iml_range[0]
            iml_range = np.arange(step, iml_max + step, step)

        iml_news_all = iml_range

        # Demolition probabilities initialization
        p_demol_final = []
        iml_demol_final = iml_news_all[iml_news_all <= iml_max]
        # Sort the maximum IMLs obtained associated with each record
        iml_max_sorted = np.sort([max(ida_results['IDA'][record]['IM']) for record in ida_results['IDA'] if
                                  len(ida_results['IDA'][record]) != 0])

        iml_stop = iml_max_sorted[-3]
        # TODO, check if vectorization may be applied instead of the for loops for the reduction of computational time
        for iml_test in iml_news_all:

            if iml_test <= iml_max:
                # Stop if limit IML is reached
                if iml_test > iml_stop:
                    iml_test = iml_stop
                # Vector to populate if the limit state is exceeded
                exceeds = []
                for record in [key for key in ida_results['IDA'].keys() if len(ida_results['IDA'][key].keys()) != 0]:
                    edp = ida_results['IDA'][record][edp_to_process]
                    iml = ida_results['IDA'][record]['IM']
                    edp = np.array(edp)
                    iml = np.array(iml)
                    edp = np.insert(edp, 0, 0)
                    iml = np.insert(iml, 0, 0)
                    order = iml.argsort()
                    edp = edp[order]
                    iml = iml[order]
                    spline = interp1d(iml, edp)
                    try:
                        edp_exceed = spline(iml_test)
                        exceeds.append(edp_exceed)
                    except:
                        pass
                edp_min = min(exceeds)
                edp_max = max(exceeds)
                num_recs = len(exceeds)
                counts = []
                edp_news = np.linspace(0, edp_max*1.5, 200)[1:]
                for edp_new in edp_news:
                    if edp_new < edp_min:
                        count = 0
                    elif edp_new > edp_max:
                        count = len(exceeds)
                    else:
                        count = sum(np.array(exceeds) < edp_new)
                    counts.append(count)
                xs = edp_news
                ys = counts
                if xs[0] == 0:
                    xs, ys = xs[1:], counts[1:]
                theta_hat_mom = np.exp(np.mean(np.log(xs)))
                beta_hat_mom = np.std(np.log(xs))
                x0 = [theta_hat_mom, beta_hat_mom]
                xopt_mle = optimize.fmin(func=lambda var: self.mlefit(theta=[var[0], var[1]], num_recs=num_recs,
                                                                      num_collapse=np.array(ys), IM=xs),
                                         x0=x0, maxiter=100, maxfun=100, disp=False)
                theta_mle = xopt_mle[0]
                beta_mle = xopt_mle[1]
                p_demol_iml = stats.norm.cdf(np.log(theta_mle/ls_median)/(beta_mle**2 + ls_cov**2)**0.5, loc=0, scale=1)

                p_demol_final.append(p_demol_iml)
            else:
                pass
        # Final fitting
        xs = iml_demol_final
        ys = [round(i, 0) for i in np.array(p_demol_final)*num_recs]

        theta_hat_mom = np.exp(np.mean(np.log(xs)))
        beta_hat_mom = np.std(np.log(xs))
        x0 = [theta_hat_mom, beta_hat_mom]
        xopt_mle = optimize.fmin(func = lambda var: self.mlefit(theta=[var[0], var[1]], num_recs=num_recs,
                                                                num_collapse=np.array(ys), IM=xs),
                                 x0=x0, maxiter=100, maxfun=100, disp=False)
        theta_mle = xopt_mle[0]
        beta_mle = xopt_mle[1]
        return {'theta': theta_mle, 'beta': beta_mle}

    def calc_p_edp_given_im(self, results, story, edp_to_process, iml_test, edp_range):
        """
        This function calculates a EDPvsPoE fragility for a given IM level, in terms of the PDF.
        Collapsed records are ignored. When less than 3 have not collapsed, the distribution at the previous IML is
        assumed
        :param results: dict                    IDA outputs
        :param story: int                       Story level under consideration
        :param edp_to_process: str              Type of EDP under consideration
        :param iml_test: float                  IML
        :param edp_range: list                  EDP range
        :return: dict                           Mean and dispersion of the CDFs of the fragility functions derived
        """
        if edp_to_process == 'PFA':
            edp_to_process_mod = 'FA'
        else:
            edp_to_process_mod = edp_to_process

        iml_max = max([max(results['IDA'][record]['IM']) for record in results['IDA'] if
                       len(results['IDA'][record]) != 0])
        iml_max = max(iml_max, iml_test)
        iml_max_sorted = np.sort([max(results['IDA'][record]['IM']) for record in results['IDA'] if
                                  len(results['IDA'][record]) != 0])
        iml_stop = iml_max_sorted[-3]
        if iml_test > iml_stop:
            iml_test = iml_stop
        
        exceeds = []
        for record in results['summary_results'].keys():
            if len(results['summary_results'][record]) != 0:
                if len(results['summary_results'][record].keys()) > 1:
                    edp = [results['summary_results'][record][i]['max%s' % edp_to_process_mod][story] for i in
                           sorted(results['summary_results'][record].keys())]
                    iml = list(sorted(results['summary_results'][record].keys()))
                    edp = np.array(edp)
                    iml = np.array(iml)
                    edp = np.insert(edp, 0, 0)
                    iml = np.insert(iml, 0, 0)
                    order = iml.argsort()
                    edp = edp[order]
                    iml = iml[order]
                    spline = interp1d(iml, edp)
                    try:
                        edp_exceed = spline(iml_test)
                        exceeds.append(edp_exceed)
                    except:
                        pass
        edp_min = min(exceeds)
        edp_max = max(exceeds)
        num_recs = len(exceeds)
        edp_news = edp_range
        counts = []
        edp_news = np.linspace(0, edp_max*1.5, 200)[1:]
        for edp_new in edp_news:
            if edp_new < edp_min:
                count = 0
            elif edp_new > edp_max:
                count = len(exceeds)
            else:
                count = sum(np.array(exceeds) < edp_new)
            counts.append(count)
        xs = edp_news
        ys = counts
        if xs[0] == 0:
            xs, ys = xs[1:], counts[1:]
        theta_hat_mom = np.exp(np.mean(np.log(xs)))
        beta_hat_mom = np.std(np.log(xs))
        x0 = [theta_hat_mom, beta_hat_mom]
        xopt_mle = optimize.fmin(func=lambda var: self.mlefit(theta=[var[0], var[1]], num_recs=num_recs,
                                                              num_collapse=np.array(ys), IM=xs),
                                 x0=x0, maxiter=100, maxfun=100, disp=False)
        theta_mle = xopt_mle[0]
        beta_mle = xopt_mle[1]

        return {'theta': theta_mle, 'beta': beta_mle}
