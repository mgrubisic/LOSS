import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d
from scipy import stats


class Fitting:
    def __init__(self):
        # TODO, make sure that fitting functions for all of them are working correctly. Mlefit is very sensitive to the
        #  size, content of variables.
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
        likelihood[likelihood == 0] = 1e-290
        loglik = -sum(np.log10(likelihood))
        return loglik

    def calc_collapse_fragility(self, nrha, nstories, iml_range, flat_slope=.1, use_edp_max=10.,
                                use_beta_MDL=.15):
        """
        Calculate a IMLvsPoE fragility for the collapse limit state
        :param nrha: dict               IDA outputs
        :param nstories: int            Number of stories
        :param iml_range: array         IML range
        :param flat_slope: float        Flattening slope
        :param use_edp_max: float       EDP max to use
        :param use_beta_MDL: float      Standard deviation accounting for modelling uncertainty to use
        :return: dict                   Fitting function parameters
        """
        # Take max EDPs of both directions
        try:
            nrhaKeys = nrha.keys()
            nrhaMax = np.maximum(nrha[nrhaKeys[0]], nrha[nrhaKeys[1]])
        except:
            nrhaMax = nrha[next(iter(nrha))]

        # Get number of variables per each EDP type
        # NRHA array has first columns as PFAs, and then PSDs
        npfa = nstories + 1
        # Slice for PSDs only for collapse fragility computation
        nrha = nrhaMax[:, :, npfa:]

        # Number of realizations
        nr = nrha.shape[1]

        # Get max PSDs for the structure and shrink one axis of NRHA
        nrha = np.max(nrha, axis=2)

        # Initialize
        edp_max = np.max(nrha) if np.max(nrha) < 10. else 10.
        iml_max = np.max(iml_range)

        edp_max_mod = min(edp_max, use_edp_max)
        iml_max_mod = iml_max
        exceeds = np.array([])
        flats = np.array([])

        for rec in range(nr):
            # Get IML order
            try:
                order = iml_range[:, rec].argsort()
                imlRec = iml_range[:, rec][order]
            except:
                order = iml_range.argsort()
                imlRec = iml_range[order]
            imlRec = np.append(imlRec, imlRec[-1])

            # Get EDP range for each simulation/record
            edp = nrha[:, rec]
            # Sort EDPs by IML order
            edp = edp[order]
            # Flatline
            edp = np.append(edp, edp_max_mod)

            # Create a spline of edp vs iml
            spline = self.spline(edp, imlRec)
            edp_news = spline["edp_spline"]
            iml_news = spline["iml_spline"]
            slope_init = imlRec[1] / edp[1]
            slopes = np.diff(iml_news) / np.diff(edp_news)
            try:
                flat_idx = np.where(slopes == slopes[(slopes < flat_slope * slope_init) & (slopes > 0) &
                                                     (slopes != np.inf)][0])[0][0]
            except:
                flat_idx = len(iml_news) - 1
                print(f"[WARNING] IDA for record {rec} not flatlining")

            flat_iml = iml_news[flat_idx - 1]
            flat_edp = edp_news[flat_idx - 1]
            flat_edp_lim = use_edp_max
            if flat_edp > flat_edp_lim:
                flat_edp = flat_edp_lim
                flat_iml = float(iml_news[np.where(edp_news == edp_news[edp_news > flat_edp_lim][0])[0]])
            exceeds = np.append(exceeds, flat_iml)
            flats = np.append(flats, flat_iml)

        iml_min = min(exceeds)
        iml_max = max(exceeds)
        # Fragility calculations with MLE fitting
        iml_news_all = np.linspace(0, iml_max_mod, 100)
        counts = np.array([])

        for iml_new in iml_news_all:
            if iml_new < iml_min:
                count = 0
            elif iml_new > iml_max:
                count = len(exceeds)
            else:
                count = sum(np.array(exceeds) < iml_new)
            counts = np.append(counts, count)

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

    def calc_demolition_fragility(self, ridr, iml_range, iml_ext, ls_median=None, ls_cov=None):
        """
        This function calculates a IMLvsPoE fragility for the demolition limit state
        :param ridr: dict                   RIDRs (IML dimension, N records)
        :param iml_range: list              IML range
        :param ls_median: float             Median EDP
        :param ls_cov: float                EDP covariance
        :return: dict                       Fitting function parameters
        """
        if ls_median is None:
            ls_median = 0.0185
        if ls_cov is None:
            ls_cov = 0.3

        # Get the maximum IML recorded
        iml_max = np.max(iml_range)

        # Update IML range if the maximum selected is lower than the actual values obtained
        if max(iml_ext) <= iml_max:
            step = iml_ext[1] - iml_ext[0]
            iml_ext = np.arange(step, iml_max + step, step)

        iml_news_all = iml_ext
        # Demolition probabilities initialization
        p_demol_final = np.array([])
        iml_demol_final = iml_news_all[iml_news_all <= iml_max]

        num_recs = ridr.shape[1]
        for iml_test in iml_news_all:

            if iml_test <= iml_max:
                # Vector to populate if the limit state is exceeded
                exceeds = np.array([])
                for rec in range(num_recs):
                    # Get IML range recorded
                    try:
                        imlRec = np.insert(iml_range[:, rec], 0, 0)
                    except:
                        imlRec = np.insert(iml_range, 0, 0)
                    order = imlRec.argsort()
                    imlRec = imlRec[order]

                    edp = ridr[:, rec]
                    edp = np.insert(edp, 0, 0)
                    edp = edp[order]
                    spline = interp1d(imlRec, edp)

                    try:
                        edp_exceed = float(spline(iml_test))
                        exceeds = np.append(exceeds, edp_exceed)
                    except:
                        pass
                edp_min = min(exceeds)
                edp_max = max(exceeds)

                counts = np.array([])
                edp_news = np.linspace(0, edp_max * 1.5, 200)[1:]
                for edp_new in edp_news:
                    if edp_new < edp_min:
                        count = 0
                    elif edp_new > edp_max:
                        count = len(exceeds)
                    else:
                        count = sum(np.array(exceeds) < edp_new)
                    counts = np.append(counts, count)
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
                p_demol_iml = stats.norm.cdf(np.log(theta_mle / ls_median) / (beta_mle ** 2 + ls_cov ** 2) ** 0.5,
                                             loc=0, scale=1)

                p_demol_final = np.append(p_demol_final, p_demol_iml)
            else:
                pass
        # Final fitting
        xs = iml_demol_final
        ys = [round(i, 0) for i in p_demol_final * num_recs]

        theta_hat_mom = np.exp(np.mean(np.log(xs)))
        beta_hat_mom = np.std(np.log(xs))
        x0 = [theta_hat_mom, beta_hat_mom]
        xopt_mle = optimize.fmin(func=lambda var: self.mlefit(theta=[var[0], var[1]], num_recs=num_recs,
                                                              num_collapse=np.array(ys), IM=xs),
                                 x0=x0, maxiter=100, maxfun=100, disp=False)
        theta_mle = xopt_mle[0]
        beta_mle = xopt_mle[1]
        return {'theta': theta_mle, 'beta': beta_mle}

    def calc_p_edp_given_im(self, results, story, iml_test, iml_range):
        """
        This function calculates a EDPvsPoE fragility for a given IM level, in terms of the PDF.
        Collapsed records are ignored. When less than 3 have not collapsed, the distribution at the previous IML is
        assumed
        :param results: dict                    IDA outputs
        :param story: int                       Story level under consideration
        :param iml_test: float                  IML
        :param iml_range: array                 Original IML range
        :return: dict                           Mean and dispersion of the CDFs of the fragility functions derived
        """

        # Append zeros to the beginning of NRHA outputs to signify the IML=0 state
        nrha = np.zeros((results.shape[0] + 1, results.shape[1], results.shape[2]))
        nrha[1:, :, :] = results

        # IML range
        try:
            iml = np.zeros((iml_range.shape[0] + 1, iml_range.shape[1]))
            iml[1:, :] = iml_range
            order = np.argsort(iml)
            iml_range = np.array(list(map(lambda x, y: y[x], order, iml)))
            # Get the EDP ranges that exceed the test IML for a given storey
            edp = nrha[:, :, story - 1]
            edp = np.array(list(map(lambda x, y: y[x], order, edp)))
        except:
            iml_range = np.insert(iml_range, 0, 0)
            order = iml_range.argsort()
            iml_range = iml_range[order]
            # Get the EDP ranges that exceed the test IML for a given storey
            edp = nrha[:, :, story - 1]
            edp = edp[order]

        if iml_test > np.max(iml_range):
            iml_test = np.max(iml_range)

        try:
            spline = []
            exceeds = np.zeros((iml_range.shape[1], ))
            for rec in range(iml_range.shape[1]):
                spl = interp1d(iml_range[:, rec], edp[:, rec], fill_value=max(edp[:, rec]), axis=0, bounds_error=False)
                spline.append(spl)
                exceeds[rec] = spl(iml_test)
        except:
            spline = interp1d(iml_range, edp, axis=0)
            exceeds = spline(iml_test)

        edp_max = max(exceeds)
        num_recs = len(exceeds)
        edp_news = np.linspace(0, edp_max * 1.5, 100)[1:]
        counts = sum(map(lambda x: x < edp_news, exceeds)) / num_recs

        xs = edp_news
        ys = counts
        if xs[0] == 0:
            xs, ys = xs[1:], counts[1:]

        def log_dist(xs, theta, beta):
            prob = stats.norm.cdf((np.log(xs/theta)) / beta)
            return prob

        coef_opt, coef_cov = optimize.curve_fit(log_dist, xs, ys, maxfev=10**6)
        theta_mle, beta_mle = coef_opt

        return {'theta': theta_mle, 'beta': beta_mle}
