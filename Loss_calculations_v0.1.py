import os
import subprocess
import numpy as np
import pickle
import re
import scipy
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from scipy.integrate import trapz
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.cm as cm
from pathlib import Path


import warnings
warnings.filterwarnings('ignore')

#--- Documenting duration of run
import timeit
start_time = timeit.default_timer()

tick_fontsize = 16; axis_label_fontsize = 18; legend_fontsize = 16 

def plot_as_emf(figure, **kwargs):
    inkscape_path = kwargs.get('inkscape', "C://Program Files//Inkscape//inkscape.exe")
    filepath = kwargs.get('filename', None)
    if filepath is not None:
        path, filename = os.path.split(filepath)
        filename, extension = os.path.splitext(filename)
        svg_filepath = os.path.join(path, filename+'.svg')
        emf_filepath = os.path.join(path, filename+'.emf')
        figure.savefig(svg_filepath, bbox_inches='tight', format='svg')
        subprocess.call([inkscape_path, svg_filepath, '--export-emf', emf_filepath])
        os.remove(svg_filepath)
    return

def _calc_SaT1_475_mean(ida_pickle, response_spectra_path, periods_path):
    """
    This function gets the mean SaT1 of the building under study for a return
    period of 475 years (i.e. ULS).
    """
    response_spectra = pd.read_pickle(response_spectra_path)
    periods = pd.read_csv(periods_path, index_col  = None)
#    ida_pickle_mod = ida_pickle.split('\\')[-1]
#    model_tag = ida_pickle_mod.split('__')[0].replace('_IDA_', '')
#    variant_tag = ida_pickle_mod.split('__')[1].replace('.pickle', '')
#    try: T1 = float(periods[(periods['frame'] == model_tag) & (periods['variant'] == variant_tag)]['period'])
#    except: T1 = float(periods[(periods['frame'] == model_tag) & (periods['variant'] == variant_tag.replace('N1', 'N0'))]['period'])    
    case = 1
    T1 = float(periods.iloc[case-1])
    SaT1 = float(response_spectra[response_spectra['T1'].round(2) == round(T1, 2)].loc[:, response_spectra.columns != 'T1'].mean(axis = 1))
    return SaT1

def _calc_IDA_PGA(ida_pickle, response_spectra_path, periods_path):
    """
    This function calculates the PGA associated with each IML of each record
    and stores it in the IDA results pickle (at floor 0 in 'summary_results').
    """
    response_spectra = pd.read_pickle(response_spectra_path)
    periods = pd.read_csv(periods_path, index_col  = None)
    pickle_file = open(ida_pickle,'rb'); results = pickle.load(pickle_file); pickle_file.close()
    
    T1 = float(periods.iloc[case-1])
    results_mod = results.copy()
    for record in results['summary_results'].keys():
        PGA_scaled = float(response_spectra[response_spectra['T1'] == 0.00][record])    
        SaT1 = float(response_spectra[response_spectra['T1'].round(2) == round(T1, 2)][record])            
        for iml in sorted(results['summary_results'][record].keys()):
            IDAfact = round(float(iml)/SaT1, 5)
            PGA_scaled_IDA = IDAfact*PGA_scaled
            results_mod['summary_results'][record][iml]['maxFA'][0] = PGA_scaled_IDA
    return results_mod

def _spline(edp, iml):
    i = np.arange(len(edp))
    interp_i = np.linspace(0, i.max(), 2000*i.max())
    xi = interp1d(i, edp)(interp_i)
    yi = interp1d(i, iml)(interp_i)
    return {'edp_spline': xi, 'iml_spline': yi}

# Used to give collapse fragility
# Calculates EDPvsPoE fragility
# Demolition fragility
def _mlefit(theta, num_recs, num_collapse, IM):
    """
    This function performs a lognormal CDF fit to fragility datapoints based
    on maximum likelihood method.
    """   
    p = stats.norm.cdf(np.log(IM), loc = np.log(theta[0]), scale = theta[1])
    likelihood = stats.binom.pmf(num_collapse, num_recs, p)
    loglik = -sum(np.log10(likelihood))
    return loglik

def _calc_gama(RP):
    """
    This function calculates the SaT1(RP=475y) modifier for a given return
    period.
    """  
    prob_50 = np.exp(-50/RP)*(np.exp(50/RP) - 1)
    gama = (prob_50/0.1)**(-1/3)
    return gama

def _calc_repair_cost(edp, max_cost, alfa, beta, gama, delta, epsilon):
    cost = max_cost*(epsilon*(edp**alfa)/(beta**alfa + edp**alfa) + (1-epsilon)*(edp**gama)/(delta**gama + edp**gama))
    return cost

def _calc_collapse_fragility(ida_results, use_flatenning_slope = 0.2, use_isdr_max = 10/100, use_beta_MDL = 0.2):
    """
    This function calculates a IMLvsPoE fragility for the collapse limit 
    state.
    """    
    edp_to_process = 'ISDR'
    # Calculate and plot limit state exceedance points on IDA
    edp_max = max([max(ida_results['IDA'][record][edp_to_process]) for record in ida_results['IDA'] if len(ida_results['IDA'][record]) != 0])
    iml_max = max([max(ida_results['IDA'][record]['IM']) for record in ida_results['IDA'] if len(ida_results['IDA'][record]) != 0])
    edp_max_mod = edp_max*2
    iml_max_mod = iml_max
    exceeds = []
    flats = []
    for record in ida_results['IDA']:
        if len(ida_results['IDA'][record]) != 0:
            iml = ida_results['IDA'][record]['IM']
            edp = ida_results['IDA'][record][edp_to_process]
            edp = np.array(edp)
            iml = np.array(iml)
            order = iml.argsort()
            edp = edp[order]
            iml = iml[order]
            edp = np.append(edp, edp_max_mod)
            iml = np.append(iml, iml[-1])
            spline = _spline(edp, iml)
            edp_news = spline['edp_spline']
            iml_news = spline['iml_spline']
            slope_ratio = use_flatenning_slope
            slope_init = iml[1]/edp[1]
            slopes = np.diff(iml_news)/np.diff(edp_news)
            flat_idx = np.where(slopes == slopes[(slopes < slope_ratio*slope_init) & (slopes > 0) & (slopes != np.inf)][0])[0][0]
            flat_iml = iml_news[flat_idx - 1]
            flat_edp = edp_news[flat_idx - 1] 
            flat_edp_lim = use_isdr_max
            if flat_edp > flat_edp_lim:
                flat_edp = flat_edp_lim 
                flat_iml = iml_news[np.where(edp_news == edp_news[edp_news > flat_edp_lim][0])[0]] 
            exceeds.append(flat_iml)
            flats.append((record, flat_iml))
    iml_min = min(exceeds); iml_max = max(exceeds)
    # Fragility calculations with MLE fitting
    iml_news_all = np.linspace(0, iml_max_mod, 100)
    counts = []
    for iml_new in iml_news_all: 
        if iml_new < iml_min: count = 0
        elif iml_new > iml_max: count = len(exceeds)
        else: count = sum(np.array(exceeds) < iml_new)
        counts.append(count) 
    num_recs = len(exceeds)
    xs = iml_news_all; ys = counts
    if xs[0] == 0: xs, ys = xs[1:], counts[1:] 
    theta_hat_mom = np.exp(np.mean(np.log(xs))); beta_hat_mom = np.std(np.log(xs))
    x0 = [theta_hat_mom, beta_hat_mom]
    xopt_mle = scipy.optimize.fmin(func = lambda var: _mlefit(theta = [var[0], var[1]], num_recs = num_recs, num_collapse = np.array(ys[1:]), IM = xs[1:]), x0 = x0, maxiter = 3000, maxfun  = 3000, disp = False)
    theta_mle = xopt_mle[0]
    # Applying BetaTOT from Tzimas et al 2016
    # BetaRTR from IDA "+" BetaMDL
    beta_mle = (xopt_mle[1]**2 + use_beta_MDL**2)**0.5     
    return {'theta': theta_mle, 'beta': beta_mle, 'flats': flats}

def _calc_demolition_fragility(results, iml_range):
    """
    This function calculates a IMLvsPoE fragility for the demolition limit 
    state.
    """    
    edp_to_process = 'RISDR'
    edp_max = max([max(results['IDA'][record][edp_to_process]) for record in results['IDA'] if len(results['IDA'][record]) != 0])
    iml_max = max([max(results['IDA'][record]['IM']) for record in results['IDA'] if len(results['IDA'][record]) != 0])       
    if max(iml_range) <= iml_max:
        step = iml_range[1] - iml_range[0]
        iml_range = np.arange(step, iml_max + step, step)
    iml_news_all = iml_range
    ls_median = 0.0185
    ls_cov = 0.3 
    p_demol_final = []
    iml_demol_final = iml_news_all[iml_news_all <= iml_max]
    iml_max_sorted = np.sort([max(results['IDA'][record]['IM']) for record in results['IDA'] if len(results['IDA'][record]) != 0])
    iml_stop = iml_max_sorted[-3]
    for iml_test in iml_news_all:
        if iml_test <= iml_max:
            if iml_test > iml_stop: iml_test = iml_stop
            exceeds = []
            for record in [key for key in results['IDA'].keys() if len(results['IDA'][key].keys()) != 0]:
                edp = results['IDA'][record][edp_to_process]
                iml = results['IDA'][record]['IM']
                edp = np.array(edp); iml = np.array(iml)
                edp = np.insert(edp, 0, 0); iml = np.insert(iml, 0, 0)
                order = iml.argsort()
                edp = edp[order]; iml = iml[order]
                spline = interp1d(iml, edp)
                try:
                    edp_exceed = spline(iml_test)
                    exceeds.append(edp_exceed)  
                except:
                    pass
            edp_min = min(exceeds); edp_max = max(exceeds)
            num_recs = len(exceeds)
            counts = []        
            edp_news = np.linspace(0, edp_max*1.5, 200)[1:]
            for edp_new in edp_news:
                if edp_new < edp_min: count = 0
                elif edp_new > edp_max: count = len(exceeds)
                else: count = sum(np.array(exceeds) < edp_new)
                counts.append(count)
            xs = edp_news; ys = counts
            if xs[0] == 0: xs, ys = xs[1:], counts[1:] 
            theta_hat_mom = np.exp(np.mean(np.log(xs))); beta_hat_mom = np.std(np.log(xs))
            x0 = [theta_hat_mom, beta_hat_mom]
            xopt_mle = scipy.optimize.fmin(func = lambda var: _mlefit(theta = [var[0], var[1]], num_recs = num_recs, num_collapse = np.array(ys), IM = xs), x0 = x0, maxiter = 100, maxfun = 100, disp = False)
            theta_mle = xopt_mle[0]; beta_mle = xopt_mle[1] 
            p_demol_iml = stats.norm.cdf(np.log(theta_mle/ls_median)/(beta_mle**2 + ls_cov**2)**0.5, loc = 0, scale = 1)
            p_demol_final.append(p_demol_iml)
        else:
            pass    
    # Final fitting
    xs = iml_demol_final; ys = [round(i,0) for i in np.array(p_demol_final)*num_recs]
    theta_hat_mom = np.exp(np.mean(np.log(xs))); beta_hat_mom = np.std(np.log(xs))
    x0 = [theta_hat_mom, beta_hat_mom]
    xopt_mle = scipy.optimize.fmin(func = lambda var: _mlefit(theta = [var[0], var[1]], num_recs = num_recs, num_collapse = np.array(ys), IM = xs), x0 = x0, maxiter = 100, maxfun = 100, disp = False)
    theta_mle = xopt_mle[0]; beta_mle = xopt_mle[1] 
    return {'theta': theta_mle, 'beta': beta_mle}

def _calc_p_edp_given_im(results, storey, edp_to_process, iml_test, edp_range):
    """
    This function calculates a EDPvsPoE fragility for a given IM level, in 
    terms of the PDF.
    Collapsed records are ignored. When less than 3 have not collapsed, the 
    distribution at the previous IML is assumed
    """   
    if edp_to_process == 'PFA': edp_to_process_mod = 'FA'
    else: edp_to_process_mod = edp_to_process  
    iml_max = max([max(results['IDA'][record]['IML']) for record in results['IDA'] if len(results['IDA'][record]) != 0])
    iml_max = max(iml_max, iml_test)
    iml_max_sorted = np.sort([max(results['IDA'][record]['IML']) for record in results['IDA'] if len(results['IDA'][record]) != 0])
    iml_stop = iml_max_sorted[-3]
    if iml_test > iml_stop: iml_test = iml_stop
    exceeds = []
    for record in results['summary_results'].keys():
        if len(results['summary_results'][record]) != 0:
            if len(results['summary_results'][record].keys()) > 1:
                edp = [results['summary_results'][record][iml]['max%s' % (edp_to_process_mod)][storey] for iml in sorted(results['summary_results'][record].keys())]
                iml = list(sorted(results['summary_results'][record].keys()))
                edp = np.array(edp); iml = np.array(iml)
                edp = np.insert(edp, 0, 0); iml = np.insert(iml, 0, 0)
                order = iml.argsort()
                edp = edp[order]; iml = iml[order]
                spline = interp1d(iml, edp)
                try:
                    edp_exceed = spline(iml_test)
                    exceeds.append(edp_exceed)  
                except:
                    pass
    edp_min = min(exceeds); edp_max = max(exceeds)
    num_recs = len(exceeds)
    edp_news = edp_range     
    counts = []        
    edp_news = np.linspace(0, edp_max*1.5, 200)[1:]
    for edp_new in edp_news:
        if edp_new < edp_min: count = 0
        elif edp_new > edp_max: count = len(exceeds)
        else: count = sum(np.array(exceeds) < edp_new)
        counts.append(count)
    xs = edp_news; ys = counts
    if xs[0] == 0: xs, ys = xs[1:], counts[1:] 
    theta_hat_mom = np.exp(np.mean(np.log(xs))); beta_hat_mom = np.std(np.log(xs))
    x0 = [theta_hat_mom, beta_hat_mom]
    xopt_mle = scipy.optimize.fmin(func = lambda var: _mlefit(theta = [var[0], var[1]], num_recs = num_recs, num_collapse = np.array(ys), IM = xs), x0 = x0, maxiter = 100, maxfun = 100, disp = False)
    theta_mle = xopt_mle[0]; beta_mle = xopt_mle[1]
    return {'theta': theta_mle, 'beta': beta_mle}   

def _calc_losses(ida_pickle, results, iml_range, option): 
#    if 'MRF' not in ida_pickle.split('\\')[-1]: nstoreys = int(ida_pickle.split('_CHS')[0].split('_')[-1])
#    else: nstoreys = int(ida_pickle.split('_MRF_')[1].split('_')[0])
    nstoreys = 5
    # Expected Losses Calculation
    ## Generate dataframe to store loss results
    df_headers = ['C', 'D']
    df_headers_storey = ['_R_ISDR_SD', '_R_ISDR_NSD', '_R_ISDR_TOTAL', '_R_PFA', '_R_TOTAL']
    for storey in np.arange(1, nstoreys + 1, 1):
        for idx in range(len(df_headers_storey)):
            df_headers.append('%s%s' % (storey, df_headers_storey[idx]))
    df_headers += ['R_ISDR_SD_TOTAL', 'R_ISDR_NSD_TOTAL', 'R_ISDR_TOTAL_TOTAL', 'R_PFA_TOTAL', 'R_TOTAL_TOTAL']
    loss_results = pd.DataFrame(columns = df_headers, index = ['%.2f' % (i) for i in iml_range])
    ## Collapse losses
    frag_calc = _calc_collapse_fragility(results); theta_col = frag_calc['theta']; beta_col = frag_calc['beta']
    p_collapse = stats.norm.cdf(np.log(iml_range/theta_col)/beta_col, loc = 0, scale = 1)
    loss_results['C'] = p_collapse
    ## Demolition losses given no collapse - P(D|NC,IM)
    frag_calc = _calc_demolition_fragility(results, iml_range); theta_dem = frag_calc['theta']; beta_dem = frag_calc['beta']
    p_demol = stats.norm.cdf(np.log(iml_range/theta_dem)/beta_dem, loc = 0, scale = 1)
    loss_results['D'] = p_demol
    ## Repair losses
    idr = np.linspace(0,0.20,1001)[1:]
    acc = np.linspace(0,20,1001)[1:]
    if option == 'HAZUS':
        if True in [bool(re.search(i, str(ida_pickle))) for i in ['_X_','_SD_','_V_','_IV_', '_V_IV_']]: system_type = 'BF'
        else: system_type = 'MF'
        # Ratios of building value for each component type
        User_sd = 0.25      # Structural Drift Sensitive
        User_nsd = 0.55     # Non-Structural Drift Sensitive
        User_acc = 0.20     # Non-Structural Acceleration Sensitive
        # Damage curves HAZUS
        ## Structural components
        ### Median [-] (Table 5.9a)
        if system_type == 'MF':
            if nstoreys <= 3: mediana_sd = [0.0060, 0.0120, 0.030, 0.080]
            elif nstoreys <= 7: mediana_sd = [0.0040, 0.0080, 0.0200, 0.0533]
            else: mediana_sd = [0.0030, 0.0060, 0.0150, 0.0400]
        elif system_type == 'BF':
            if nstoreys <=3 : mediana_sd = [0.0050, 0.0100, 0.030, 0.080]
            elif nstoreys <= 7: mediana_sd = [0.0033, 0.0067, 0.0200, 0.0533]
            else: mediana_sd = [0.0025, 0.0050, 0.0150, 0.0400]
        ### COV
        beta_sd = 0.5
        ## Non-structural components - Drift sensitive
        ### Median [-] (Table 5.10)
        mediana_nsd = [0.0040, 0.0080, 0.025, 0.05]
        ### COV
        beta_nsd = 0.50
        ## Non-structural components - Acceleration sensitive
        ### Median [g] (Table 5.12, High-Code)
        mediana_acc = [0.30, 0.60, 1.20, 2.40]
        ### COV
        beta_acc = 0.60
        ## CDF's Damage - 4 Damage Limit States
        ### Structural Drift Sensitive
        y1_sd = stats.norm.cdf(np.log(idr/mediana_sd[0])/beta_sd, loc = 0, scale = 1)
        y2_sd = stats.norm.cdf(np.log(idr/mediana_sd[1])/beta_sd, loc = 0, scale = 1)
        y3_sd = stats.norm.cdf(np.log(idr/mediana_sd[2])/beta_sd, loc = 0, scale = 1)
        y4_sd = stats.norm.cdf(np.log(idr/mediana_sd[3])/beta_sd, loc = 0, scale = 1)
        ### Non-Structural Drift Sensitive
        y1_nsd = stats.norm.cdf(np.log(idr/mediana_nsd[0])/beta_nsd, loc = 0, scale = 1)
        y2_nsd = stats.norm.cdf(np.log(idr/mediana_nsd[1])/beta_nsd, loc = 0, scale = 1)
        y3_nsd = stats.norm.cdf(np.log(idr/mediana_nsd[2])/beta_nsd, loc = 0, scale = 1)
        y4_nsd = stats.norm.cdf(np.log(idr/mediana_nsd[3])/beta_nsd, loc = 0, scale = 1)
        ### Non-Structural Acceleration Sensitive
        y1_acc = stats.norm.cdf(np.log(acc/mediana_acc[0])/beta_acc, loc = 0, scale = 1)
        y2_acc = stats.norm.cdf(np.log(acc/mediana_acc[1])/beta_acc, loc = 0, scale = 1)
        y3_acc = stats.norm.cdf(np.log(acc/mediana_acc[2])/beta_acc, loc = 0, scale = 1)
        y4_acc = stats.norm.cdf(np.log(acc/mediana_acc[3])/beta_acc, loc = 0, scale = 1)
        ## Probability Associate With Each Damage Limit State
        ### Structural Drift Sensitive
        Pds4_sd = y4_sd
        Pds3_sd = y3_sd - y4_sd
        Pds2_sd = y2_sd - y3_sd
        Pds1_sd = y1_sd - y2_sd
        ### Non-Structural Drift Sensitive
        Pds4_nsd = y4_nsd
        Pds3_nsd = y3_nsd - y4_nsd
        Pds2_nsd = y2_nsd - y3_nsd
        Pds1_nsd = y1_nsd - y2_nsd
        ### Non-Structural Acceleration Sensitive
        Pds4_acc = y4_acc
        Pds3_acc = y3_acc - y4_acc
        Pds2_acc = y2_acc - y3_acc
        Pds1_acc = y1_acc - y2_acc
        # Repair cost ratios
        ## Repair costs Associated with Multifamiliar Dweling Building - HAZUS
        ### Structural Drift Sensitive (Table 15.2)
        custos_struct = [0.003, 0.014, 0.069, 0.138]
        ### Non-Structural Drift Sensitive (Table 15.4)
        custos_nonstructIDR = [0.009, 0.043, 0.213, 0.425]
        ### Non-Structural Acceleration Sensitive (table 15.3)
        custos_nonstructPFA = [0.008, 0.043, 0.131, 0.437]
        ## Costs sorted by group of elements
        ### Structural Drift Sensitive
        total_sd = Pds1_sd*custos_struct[0] + Pds2_sd*custos_struct[1] + Pds3_sd*custos_struct[2] + Pds4_sd*custos_struct[3]
        ### Non-Structural Drift Sensitive
        total_nsd = Pds1_nsd*custos_nonstructIDR[0] + Pds2_nsd*custos_nonstructIDR[1] + Pds3_nsd*custos_nonstructIDR[2] + Pds4_nsd*custos_nonstructIDR[3]
        ### Non-Structural Acceleration Sensitive
        total_acc = Pds1_acc*custos_nonstructPFA[0] + Pds2_acc*custos_nonstructPFA[1] + Pds3_acc*custos_nonstructPFA[2] + Pds4_acc*custos_nonstructPFA[3]
        ## Normalized Costs
        ### Structural Drift Sensitive
        total_sd_scaled = total_sd/max(total_sd)
        ### Non-Structural Drift Sensitive
        total_nsd_scaled = total_nsd/max(total_nsd)
        ### Non-Structural Acceleration Sensitive
        total_acc_scaled = total_acc/max(total_acc)
        ## Re-define weight 
        ### Structural Drift Sensitive
        W_sd = User_sd
        ### Non-Structural Drift Sensitive
        W_nsd = User_nsd
        ### Non-Structural Acceleration Sensitive
        W_acc = User_acc
        ## Re-Scaled Costs
        ### Structural Drift Sensitive
        total_IDR_SD = W_sd*total_sd_scaled
        ### Non-Structural Drift Sensitive
        total_IDR_NSD = W_nsd*total_nsd_scaled
        ### Non-Structural Acceleration Sensitive
        total_PFA = W_acc*total_acc_scaled
        ## Re-Scaled Cost splines
        ### Structural Drift Sensitive
        total_IDR_SD_spline = interp1d(np.insert(idr, 0, 0), np.insert(total_IDR_SD, 0, 0))
        ### Non-Structural Drift Sensitive
        total_IDR_NSD_spline = interp1d(np.insert(idr, 0, 0), np.insert(total_IDR_NSD, 0, 0))
        ### Non-Structural Acceleration Sensitive
        total_PFA_spline = interp1d(np.insert(acc, 0, 0), np.insert(total_PFA, 0, 0))
        ## Repair losses
        storey_loss_ratio_weights = [1/nstoreys for i in range(nstoreys)]
        for iml in iml_range:
            iml_test = iml
            r_isdr_sd_total = 0
            r_isdr_nsd_total = 0
            r_isdr_total_total = 0
            r_pfa_total = 0
            r_total_total = 0 
            for storey in np.arange(1, nstoreys + 1, 1):
                ### Drift-sensitive losses
                edp_to_process = 'ISDR'
                frag_calc = _calc_p_edp_given_im(results, storey, edp_to_process, iml, idr)
                edp_theta = frag_calc['theta']
                edp_beta = frag_calc['beta']
                p_edp = stats.norm.pdf(np.log(idr/edp_theta)/edp_beta, loc = 0, scale = 1); p_edp = p_edp/sum(p_edp)
                storey_loss_ratio_idr_sd = sum(total_IDR_SD_spline(idr)*p_edp)*storey_loss_ratio_weights[storey - 1]
                storey_loss_ratio_idr_nsd = sum(total_IDR_NSD_spline(idr)*p_edp)*storey_loss_ratio_weights[storey - 1]
                ### Acceleration-sensitive losses
                edp_to_process = 'PFA'
                storey_loss_ratio_acc = 0
                for floor in [storey - 1, storey]:
                    frag_calc = _calc_p_edp_given_im(results, floor, edp_to_process, iml, acc)
                    edp_theta = frag_calc['theta']
                    edp_beta = frag_calc['beta']
                    p_edp = stats.norm.pdf(np.log(acc/edp_theta)/edp_beta, loc = 0, scale = 1); p_edp = p_edp/sum(p_edp)
                    storey_loss_ratio_acc_partial = sum(total_PFA_spline(acc)*p_edp)
                    storey_loss_ratio_acc_partial /= 2
                    storey_loss_ratio_acc += storey_loss_ratio_acc_partial
                storey_loss_ratio_acc *= storey_loss_ratio_weights[storey - 1]
                r_isdr_sd = storey_loss_ratio_idr_sd
                r_isdr_nsd = storey_loss_ratio_idr_nsd
                r_isdr_total = r_isdr_sd + r_isdr_nsd
                r_pfa = storey_loss_ratio_acc
                r_total = r_isdr_total + r_pfa
                columns = ['%s%s' % (storey, i) for i in df_headers_storey]
                values = [r_isdr_sd, r_isdr_nsd, r_isdr_total, r_pfa, r_total]
                loss_results.loc['%.2f' % (iml_test), columns] = values
                r_isdr_sd_total += r_isdr_sd
                r_isdr_nsd_total += r_isdr_nsd
                r_isdr_total_total += r_isdr_total
                r_pfa_total += r_pfa
                r_total_total += r_total
                columns = ['R_ISDR_SD_TOTAL', 'R_ISDR_NSD_TOTAL','R_ISDR_TOTAL_TOTAL', 'R_PFA_TOTAL', 'R_TOTAL_TOTAL']
                values = [r_isdr_sd_total, r_isdr_nsd_total, r_isdr_total_total, r_pfa_total, r_total_total]
                loss_results.loc['%.2f' % (iml_test), columns] = values 
        loss_results['E_NC_ND_ISDR_S'] = loss_results['R_ISDR_SD_TOTAL']*(1-loss_results['C'])*(1-loss_results['D'])
        loss_results['E_NC_ND_ISDR_NS'] = loss_results['R_ISDR_NSD_TOTAL']*(1-loss_results['C'])*(1-loss_results['D'])
        loss_results['E_NC_ND_ISDR_TOTAL'] = loss_results['R_ISDR_TOTAL_TOTAL']*(1-loss_results['C'])*(1-loss_results['D'])
        loss_results['E_NC_ND_PFA_TOTAL'] = loss_results['R_PFA_TOTAL']*(1-loss_results['C'])*(1-loss_results['D'])
        loss_results['E_NC_ND_S'] = loss_results['E_NC_ND_ISDR_S']
        loss_results['E_NC_ND_NS'] = loss_results['E_NC_ND_ISDR_NS'] + loss_results['E_NC_ND_PFA_TOTAL']
        loss_results['E_NC_ND'] = loss_results['E_NC_ND_S'] + loss_results['E_NC_ND_NS']
        loss_results['E_NC_D'] = loss_results['D']*(1-loss_results['C'])
        loss_results['E_C'] = loss_results['C']
        loss_results['E_LT'] = loss_results['E_NC_ND'] + 1*loss_results['E_NC_D'] + 1*loss_results['E_C']
        
    return loss_results

def plot_edps_at_imls(ida_pickle, results, response_spectra_path, periods_path):
    def _modify_results(levels, isdr):
        levels_mod = [0]
        for idx, i in enumerate(levels):
            levels_mod.append(i)
            if idx != len(levels) - 1: levels_mod.append(i)
        isdr_mod = []
        for idx, i in enumerate(isdr):
            isdr_mod.append(i)
            isdr_mod.append(i)
        return levels_mod, isdr_mod
    sa_t1_475 = _calc_SaT1_475_mean(ida_pickle, response_spectra_path, periods_path)
    iml_sls1 = _calc_gama(95)*sa_t1_475
    iml_sls3 = _calc_gama(225)*sa_t1_475
    iml_uls = _calc_gama(475)*sa_t1_475
    iml_cls = _calc_gama(2475)*sa_t1_475
    record_0 = list(results['summary_results'].keys())[0]
    iml_0 = list(results['summary_results'][record_0].keys())[0]
    imls = [iml_sls1, iml_sls3, iml_uls, iml_cls]
    cmap = cm.get_cmap('binary')
    color_cmap = cmap(np.linspace(0.2, 1.0, len(imls)))
    marker_list = ['o', 'v', '^', 's']
    markers_select = np.random.choice(marker_list, len(imls), replace=False)
    iml_labels = ['SLS-1', 'SLS-3', 'ULS', 'CLS']
    colors = {}
    markers = {}
    labels = {}
    for idx, iml in enumerate(imls): 
        colors['%.3f' % (iml)] = color_cmap[idx]
        markers['%.3f' % (iml)] = markers_select[idx]
        labels['%.3f' % (iml)] = iml_labels[idx]
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 5))        
    for edp_to_process in ['ISDR', 'RISDR', 'PFA']:
        plot_edp_max = {'ISDR': 2.5/100, 'RISDR': 0.1/100, 'PFA': 2.5}[edp_to_process]
        plot_edp_divs = {'ISDR': 5, 'RISDR':5, 'PFA': 5}[edp_to_process]
        x_ticks = np.linspace(0, plot_edp_max, plot_edp_divs + 1)
        if edp_to_process == 'ISDR': ax = ax1
        elif edp_to_process == 'RISDR': ax = ax2
        elif edp_to_process == 'PFA': ax = ax3        
        if edp_to_process == 'PFA': edp_to_process_mod = 'FA'
        else: edp_to_process_mod = edp_to_process 
        for iml_test in imls:
            color = colors['%.3f' % (iml_test)]
            marker = markers['%.3f' % (iml_test)]
            label = '%.2f' % (iml_test) + ' (%s)' % (labels['%.3f' % (iml_test)])
            storeys = []; edps = []
            for storey in sorted(results['summary_results'][record_0][iml_0]['max%s' % (edp_to_process_mod)].keys()):
                storeys.append(storey)
                iml_max = max([max(results['IDA'][record]['IM']) for record in results['IDA'] if len(results['IDA'][record]) != 0])
                iml_max = max(iml_max, iml_test)
                exceeds = []
                for record in results['summary_results'].keys():
                    edp = [results['summary_results'][record][iml]['max%s' % (edp_to_process_mod)][storey] for iml in sorted(results['summary_results'][record].keys())]
                    iml = list(sorted(results['summary_results'][record].keys()))
                    edp = np.array(edp); iml = np.array(iml)
                    edp = np.insert(edp, 0, 0); iml = np.insert(iml, 0, 0)
                    order = iml.argsort()
                    edp = edp[order]; iml = iml[order]
                    edp = np.append(edp, edp[-1]); iml = np.append(iml, iml_max) 
                    spline = interp1d(iml, edp)
                    edp_exceed = spline(iml_test)
                    exceeds.append(edp_exceed)
                median_edp = np.median(exceeds)
                edps.append(median_edp)
            storeys_mod, edps_mod = _modify_results(storeys, edps)
            if edp_to_process == 'PFA': ax.plot(edps, storeys, label = label, color = color, marker = marker)
            else: ax.plot(edps_mod, storeys_mod, label = label, color = color, marker = marker)
        ax.set_xticks(x_ticks)
        ax.set_xlim(x_ticks[0], x_ticks[-1])
        if edp_to_process == 'ISDR': ax.set_xticklabels(['{:.1f}'.format(x*100) for x in x_ticks], fontsize = tick_fontsize)
        elif edp_to_process == 'RISDR':  ax.set_xticklabels(['{:.2f}'.format(x*100) for x in x_ticks], fontsize = tick_fontsize)
        elif edp_to_process == 'PFA': ax.set_xticklabels(['{:.1f}'.format(x) for x in x_ticks], fontsize = tick_fontsize)
        if edp_to_process in ['ISDR', 'RISDR']: ax.set_xlabel(r'$\mathrm{\overline{P%s}}$' % (edp_to_process) + ' [%]', fontsize = axis_label_fontsize)
        elif edp_to_process in ['PFA']: ax.set_xlabel(r'$\mathrm{\overline{%s}}$' % (edp_to_process) + ' [g]', fontsize = axis_label_fontsize)
        ax.grid(color = '0.75')
        y_ticks = np.arange(0, max(storeys) + 1, 1)
        ax.set_yticks(y_ticks)
        ax.set_ylim(y_ticks[0], y_ticks[-1])
        ax.set_yticklabels(y_ticks, fontsize = tick_fontsize)
    ax1.set_ylabel('Floor', fontsize = axis_label_fontsize) 
    lg = ax3.legend(fontsize = legend_fontsize, frameon = False, title = r'$\mathrm{S_a(T_1, \xi)}$ [g]', loc='center left', bbox_to_anchor=(1, 0.5))
    lg.get_title().set_fontsize(legend_fontsize)
    f.tight_layout()
    fig = plt.gcf()  
    plot_as_emf(fig, filename = '%s_edps_at_imls.emf' % (case))
    plt.close()
    
def plot_collapse_fragility(results, iml_range):
    f, ax = plt.subplots(1, 1, figsize = (5, 5)) 
    xlim_iml = 5
    ylim_poe = 1
    frag_calc = _calc_collapse_fragility(results); theta_mle = frag_calc['theta']; beta_mle = frag_calc['beta']
    p = stats.norm.cdf(np.log(iml_range/theta_mle)/beta_mle, loc = 0, scale = 1)
    label =  r'$\mathrm{\theta\ = }$'+ '%.2f' % (theta_mle) + '\n' + r'$\mathrm{\beta\ = }$' + '%.2f' % (beta_mle)
    ax.annotate(label, xy = (0.05*xlim_iml, 0.95*ylim_poe), va = 'top', ha = 'left', fontsize = legend_fontsize, color = 'b')
    ax.plot(iml_range, p, color = 'b')
    for ax in f.axes:
        x_max = xlim_iml
        ax.set_xlim(0, x_max)
        ax.set_xticks(np.linspace(0, x_max, 6))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.set_xlabel(r'$\mathrm{S_a(T_1, \xi)}$ [g]', fontsize = axis_label_fontsize) 
        ax.set_xticklabels(['{:,.1f}'.format(x) for x in ax.get_xticks()], fontsize = tick_fontsize)
        y_max = ylim_poe 
        ax.set_ylim(0, y_max)
        ax.set_yticks(np.linspace(0, y_max, 6)) 
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(length = 5)
        ax.yaxis.set_tick_params(length = 5)
        ax.grid(color = '0.75') 
        ax.set_yticklabels(['{:,.1f}'.format(x) for x in ax.get_yticks()], fontsize = tick_fontsize)
        ax.set_ylabel('P [Collapse]', fontsize = axis_label_fontsize)
    f.tight_layout()
    fig = plt.gcf()  
    plot_as_emf(fig, filename = '%s_collapse.emf' % (case))
    plt.close()

def plot_demolition_fragility(results, iml_range):
    f, ax = plt.subplots(1, 1, figsize = (5, 5)) 
    frag_calc = _calc_demolition_fragility(results, iml_range); theta_mle = frag_calc['theta']; beta_mle = frag_calc['beta']
    p = stats.norm.cdf(np.log(iml_range/theta_mle)/beta_mle, loc = 0, scale = 1)
    label =  r'$\mathrm{\theta\ = }$'+ '%.2f' % (theta_mle) + '\n' + r'$\mathrm{\beta\ = }$' + '%.2f' % (beta_mle)
    ax.plot(iml_range, p, color = 'r')
    xlim_iml = 5
    ylim_poe = 1
    ax.annotate(label, xy = (0.05*xlim_iml, 0.95*ylim_poe), va = 'top', ha = 'left', fontsize = legend_fontsize, color = 'r')
    for ax in f.axes:
        x_max = xlim_iml
        ax.set_xlim(0, x_max)
        ax.set_xticks(np.linspace(0, x_max, 6))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.set_xlabel(r'$\mathrm{S_a(T_1, \xi)}$ [g]', fontsize = axis_label_fontsize) 
        ax.set_xticklabels(['{:,.1f}'.format(x) for x in ax.get_xticks()], fontsize = tick_fontsize)
        y_max = ylim_poe 
        ax.set_ylim(0, y_max)
        ax.set_yticks(np.linspace(0, y_max, 6)) 
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(length = 5)
        ax.yaxis.set_tick_params(length = 5)
        ax.grid(color = '0.75') 
        ax.set_yticklabels(['{:,.1f}'.format(x) for x in ax.get_yticks()], fontsize = tick_fontsize)
        ax.set_ylabel('P [Demolition]', fontsize = axis_label_fontsize)
    f.tight_layout()
    fig = plt.gcf()  
    plot_as_emf(fig, filename = '%s_demolition.emf' % (case))
    plt.close()   
    
def plot_loss_curves(ida_pickle, results, iml_range, response_spectra_path, periods_path, frame_name, option, include_legend = True):
    # Calculate loss results
    loss_results = _calc_losses(ida_pickle, results, iml_range, option)
    loss_results.to_csv('%s_loss_curves.txt' % (frame_name))   
    IML = np.insert(np.array([float(i) for i in loss_results.index]), 0, 0)
    E_NC_ND_ISDR_S = np.insert(np.array(loss_results['E_NC_ND_ISDR_S']), 0, 0)
    E_NC_ND_ISDR_NS = np.insert(np.array(loss_results['E_NC_ND_ISDR_NS']), 0, 0)
    E_NC_ND_ISDR_TOTAL = np.insert(np.array(loss_results['E_NC_ND_ISDR_TOTAL']), 0, 0)
    E_NC_ND_PFA_TOTAL = np.insert(np.array(loss_results['E_NC_ND_PFA_TOTAL']), 0, 0)
    E_NC_ND_S = np.insert(np.array(loss_results['E_NC_ND_S']), 0, 0)
    E_NC_ND_NS = np.insert(np.array(loss_results['E_NC_ND_NS']), 0, 0)
    E_NC_ND = np.insert(np.array(loss_results['E_NC_ND']), 0, 0)
    E_NC_D = np.insert(np.array(loss_results['E_NC_D']), 0, 0)
    E_C = np.insert(np.array(loss_results['E_C']), 0, 0)
    E_LT = np.insert(np.array(loss_results['E_LT']), 0, 0) 
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
    # Calculate collapse risk
    #### UPDATE PATH
    hazard_path = response_spectra_path
    location = frame_name.split('_')[0]
    T_hazard = list(pd.read_excel('%s\\Hazard.xlsx' % (hazard_path), sheet_name = '%s_T' % (location), header = None)[0])
    IMLs_hazard = pd.read_excel('%s\\Hazard.xlsx' % (hazard_path), sheet_name = '%s_IMLs' % (location), header = None, names = T_hazard)
    Probs_hazard = pd.read_excel('%s\\Hazard.xlsx' % (hazard_path), sheet_name = '%s_Probs' % (location), header = None, names = T_hazard)
    Rates_hazard = -np.log(1-Probs_hazard)/50
    T_hazard = np.array(T_hazard)
    periods = pd.read_csv(periods_path, index_col  = None)
    model_tag = frame_name.split('__')[0]
    variant_tag = ida_pickle.split('\\')[-1].split('__')[-1].replace('.pickle', '')
    try: T1 = float(periods[(periods['frame'] == model_tag) & (periods['variant'] == variant_tag)]['period'])
    except: pass
    try: T1 = float(periods[(periods['frame'] == model_tag) & (periods['variant'] == variant_tag.replace('N1', 'N0'))]['period'])
    except: pass
    try: T1 = float(periods[(periods['frame'] == model_tag) & (periods['variant'] == variant_tag.replace('N0', 'N1'))]['period'])
    except: pass
    try: T1 = float(periods[(periods['frame'] == model_tag) & (periods['variant'] == variant_tag.replace('_N0', ''))]['period'])
    except: pass
    try: T1 = float(periods[(periods['frame'] == model_tag) & (periods['variant'] == variant_tag.replace('_N1', ''))]['period'])
    except: pass
    if T1 in T_hazard:
        IML_hazard = np.array(IMLs_hazard[T1])
        Rate_hazard = np.array(Rates_hazard[T1])
    else:
        period_low = T_hazard[T_hazard <= T1][-1]
        period_high = T_hazard[T_hazard >= T1][0]
        IML_hazard = np.array(IMLs_hazard[period_low] - (period_low - T1)/((period_low - period_high)/(IMLs_hazard[period_low] - IMLs_hazard[period_high])))
        Rate_hazard = np.array(Rates_hazard[period_low] - (period_low - T1)/((period_low - period_high)/(Rates_hazard[period_low] - Rates_hazard[period_high])))    
    Rate_spline = interp1d(IML_hazard, Rate_hazard)    
    IML_hazard_step = IML_hazard[1] - IML_hazard[0]
    IML_hazard = np.arange(IML_hazard[0], IML_hazard[-1] + IML_hazard_step, IML_hazard_step)
    Rate_hazard = Rate_spline(IML_hazard)
    slopes = abs(np.gradient(Rate_hazard, IML_hazard_step))
    MAF = trapz(y=E_C_spline(IML_hazard)*slopes, x=IML_hazard)
    Pc_50 = 1-np.exp(-MAF*50)
    Risk = pd.DataFrame(data=[[MAF, Pc_50]], columns = ['MAF', 'Pc50'])
    Risk.to_csv('%s_MAF_PC50.txt' % (frame_name))
    # Calculate EALs and PVs
    EAL = trapz(y=E_LT_spline(IML_hazard)*slopes, x=IML_hazard)
    DR = 0.05
    lifespan = 50 
    loss_tags = ['NC_ND_ISDR_S', 'NC_ND_ISDR_NS', 'NC_ND_ISDR_TOTAL', 'NC_ND_PFA_TOTAL', 'NC_ND_S', 'NC_ND_NS', 'NC_ND', 'NC_D', 'C', 'LT']
    EAL_all = {i:0 for i in loss_tags}
    PV_all_1 = {i:0 for i in loss_tags}
    PV_all_2 = {i:0 for i in loss_tags}    
    for idx, i in enumerate(loss_tags):
        if idx == 0: loss_spline = E_NC_ND_ISDR_S_spline
        elif idx == 1: loss_spline = E_NC_ND_ISDR_NS_spline
        elif idx == 2: loss_spline = E_NC_ND_ISDR_TOTAL_spline
        elif idx == 3: loss_spline = E_NC_ND_PFA_TOTAL_spline           
        elif idx == 4: loss_spline = E_NC_ND_S_spline
        elif idx == 5: loss_spline = E_NC_ND_NS_spline
        elif idx == 6: loss_spline = E_NC_ND_spline     
        elif idx == 7: loss_spline = E_NC_D_spline
        elif idx == 8: loss_spline = E_C_spline
        elif idx == 9: loss_spline = E_LT_spline
        EAL = trapz(y=loss_spline(IML_hazard)*slopes, x=IML_hazard)
        PV_Macedo = EAL*((1 - np.exp(-DR*lifespan))/DR)
        PV_HwangLignos = EAL*sum([(1+DR)**-i for i in np.arange(1,lifespan+1,1)])
        EAL_all[i] = EAL
        PV_all_1[i] = PV_Macedo
        PV_all_2[i] = PV_HwangLignos   
        EAL_PV = pd.DataFrame(data=[list(EAL_all.values()), list(PV_all_1.values()), list(PV_all_2.values())], columns = EAL_all.keys(), index = ['EAL', 'PV_Macedo', 'PV_HwangLignos'])
        EAL_PV.to_csv('%s_EAL_PV.txt' % (frame_name))
    # Calculate storey repair loss contributions
    if 'MRF' not in ida_pickle.split('\\')[-1]: nstoreys = int(ida_pickle.split('_CHS')[0].split('_')[-1])
    else: nstoreys = int(ida_pickle.split('_MRF_')[1].split('_')[0])
    R_TOTAL = np.insert(np.array(loss_results['R_TOTAL_TOTAL']),0,0)
    R_TOTAL_spline = interp1d(IML, R_TOTAL)
    EAL_R_TOTAL = trapz(y=R_TOTAL_spline(IML_hazard)*slopes, x=IML_hazard)
    storey_list = []
    EAL_contrib_list = []
    for storey in range(1,nstoreys+1):
        R_storey = np.insert(np.array(loss_results['%s_R_TOTAL' % (storey)]),0,0)
        R_storey_spline = interp1d(IML, R_storey)
        EAL_storey = trapz(y=R_storey_spline(IML_hazard)*slopes, x=IML_hazard)
        EAL_contribution = EAL_storey/EAL_R_TOTAL
        storey_list.append(storey)
        EAL_contrib_list.append(EAL_contribution)
    EAL_R_contribution = pd.DataFrame({'storey': storey_list, 'EAL_R_contribution': EAL_contrib_list})
    EAL_R_contribution.to_csv('%s_EAL_R_contribution.txt' % (frame_name))
    # Plot loss vulnerabity curves  
    n_curves = 5
    cmap = cm.get_cmap('gnuplot')
    colors = cmap(np.linspace(0, 0.75, n_curves))
    xlim_iml = 5
    ylim_lr = 1
    f, ax1 = plt.subplots(1, 1, figsize = (5, 5))
    markevery = 0.1
    color_E_NC_ND_ISDR_S = colors[0]
    color_E_NC_ND_ISDR_NS = colors[1]
    color_E_NC_ND_PFA_TOTAL = colors[2]
    color_E_NC_D = colors[3]
    color_E_C = colors[4]
    ax1.plot(IML, E_NC_ND_S, color = color_E_NC_ND_ISDR_S, ls = '-', markersize = 7.5, markerfacecolor = 'w', marker = 's', markevery = markevery, label = 'Repair S')
    ax1.plot(IML, E_NC_ND_ISDR_NS, color = color_E_NC_ND_ISDR_NS, ls = '-', markersize = 7.5, markerfacecolor = 'w', marker = 'o', markevery = markevery, label = 'Repair NS-ISDR')
    ax1.plot(IML, E_NC_ND_PFA_TOTAL, color = color_E_NC_ND_PFA_TOTAL, ls = '-', markersize = 7.5, markerfacecolor = 'w', marker = 'v', markevery = markevery, label = 'Repair NS-PFA')
    ax1.plot(IML, E_NC_D, color = color_E_NC_D, ls = '--', label = 'Demolition')
    ax1.plot(IML, E_C, color = color_E_C, ls = ':', label = 'Collapse')
    ax1.plot(IML, E_LT, color = 'k', label = 'Total')
    for ax in f.axes:
        x_max = xlim_iml
        ax.set_xlim(0, x_max)
        ax.set_xticks(np.linspace(0, x_max, 6))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.set_xlabel(r'$\mathrm{S_a(T_1, \xi)}$ [g]', fontsize = axis_label_fontsize) 
        ax.set_xticklabels(['{:,.1f}'.format(x) for x in ax.get_xticks()], fontsize = tick_fontsize)
        y_max = ylim_lr 
        ax.set_ylim(0, y_max)
        ax.set_yticks(np.linspace(0, y_max, 6)) 
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(length = 5)
        ax.yaxis.set_tick_params(length = 5)
        ax.grid(color = '0.75') 
        ax.set_yticklabels(['{:,.1f}'.format(x) for x in ax.get_yticks()], fontsize = tick_fontsize)
        ax.set_ylabel('Loss ratio', fontsize = axis_label_fontsize)
        if include_legend == True: ax.legend(frameon = False, fontsize = legend_fontsize, loc='center left', bbox_to_anchor=(1, 0.5))
    fig = plt.gcf()  
    plot_as_emf(fig, filename = '%s_loss_curves.emf' % (frame_name))
    plt.close() 
    # Plot losses at intensities of interest of EC8 
    f, ax1 = plt.subplots(1, 1, figsize = (5, 5))    
    sa_t1_475 = _calc_SaT1_475_mean(ida_pickle, response_spectra_path, periods_path)
    iml_sls1 = _calc_gama(95)*sa_t1_475
    iml_sls3 = _calc_gama(225)*sa_t1_475
    iml_uls = _calc_gama(475)*sa_t1_475
    iml_cls = _calc_gama(2475)*sa_t1_475
    imls_interest = np.array([iml_sls1, iml_sls3, iml_uls, iml_cls])
    np.savetxt('%s_imls_interest.txt' % (frame_name), [imls_interest], delimiter='\n')
    idx = 1
    idx_list = []
    idx_label_list = []
    for iml in [iml_sls1, iml_sls3, iml_uls, iml_cls]:
        if iml == iml_sls1: iml_label_2 = 'SLS-1'
        elif iml == iml_sls3: iml_label_2 = 'SLS-3'
        elif iml == iml_uls: iml_label_2 = 'ULS'
        elif iml == iml_cls: iml_label_2 = 'CLS'    
        iml_label = '%.2f' % (iml)
        idx_label_list.append(iml_label)
        E_NC_ND_ISDR_S_iml = E_NC_ND_ISDR_S_spline(iml)
        E_NC_ND_ISDR_NS_iml = E_NC_ND_ISDR_NS_spline(iml)
        E_NC_ND_PFA_TOTAL_iml = E_NC_ND_PFA_TOTAL_spline(iml)
        E_NC_D_iml = E_NC_D_spline(iml)
        E_C_iml = E_C_spline(iml)
        idx_list.append(idx)       
        color_E_NC_ND_ISDR_S_iml = colors[0]
        color_E_NC_ND_ISDR_NS_iml = colors[1]
        color_E_NC_ND_PFA_TOTAL_iml = colors[2]
        color_E_NC_D_iml = colors[3]
        color_E_C_iml = colors[4]       
        if iml == iml_sls1:
            label_E_NC_ND_ISDR_S_iml = 'Repair S'
            label_E_NC_ND_ISDR_NS_iml = 'Repair NS-ISDR'
            label_E_NC_ND_PFA_TOTAL_iml = 'Repair NS-PFA'
            label_E_NC_D_iml = 'Demolition'
            label_E_C_iml = 'Collapse'
        else:
            label_E_NC_ND_ISDR_S_iml = None
            label_E_NC_ND_ISDR_NS_iml = None
            label_E_NC_ND_PFA_TOTAL_iml = None
            label_E_NC_D_iml = None
            label_E_C_iml = None
        ax1.bar(idx, E_NC_ND_ISDR_S_iml, color = color_E_NC_ND_ISDR_S_iml, width = 1, edgecolor = 'k', bottom = 0, zorder = 1000, label = label_E_NC_ND_ISDR_S_iml)
        ax1.bar(idx, E_NC_ND_ISDR_NS_iml, color = color_E_NC_ND_ISDR_NS_iml, width = 1, edgecolor = 'k', bottom = E_NC_ND_ISDR_S_iml, zorder = 1000, label = label_E_NC_ND_ISDR_NS_iml)
        ax1.bar(idx, E_NC_ND_PFA_TOTAL_iml, color = color_E_NC_ND_PFA_TOTAL_iml, width = 1, edgecolor = 'k', bottom = E_NC_ND_ISDR_S_iml + E_NC_ND_ISDR_NS_iml, zorder = 1000, label = label_E_NC_ND_PFA_TOTAL_iml)
        ax1.bar(idx, E_NC_D_iml, color = color_E_NC_D_iml, edgecolor = 'k', width = 1, bottom = E_NC_ND_ISDR_S_iml + E_NC_ND_ISDR_NS_iml + E_NC_ND_PFA_TOTAL_iml, zorder = 1000, label = label_E_NC_D_iml)
        ax1.bar(idx, E_C_iml, color = color_E_C_iml, edgecolor = 'k', width = 1, bottom = E_NC_ND_ISDR_S_iml + E_NC_ND_ISDR_NS_iml + E_NC_ND_PFA_TOTAL_iml + E_NC_D_iml, zorder = 1000, label = label_E_C_iml)    
        ax1.annotate(iml_label_2, xy = (idx, 0), xytext = (idx, 1.05), ha = 'center', va = 'center', fontsize = tick_fontsize, zorder = 2000)
        idx += 2
    for ax in f.axes:    
        ax.set_xticks(idx_list)
        ax.set_xticklabels(idx_label_list, fontsize = tick_fontsize)
        ax.set_xlabel(r'$\mathrm{S_a(T_1, \xi)}$ [g]', fontsize = axis_label_fontsize) 
        y_max = ylim_lr 
        ax.set_ylim(0, y_max)
        ax.set_yticks(np.linspace(0, y_max, 6)) 
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_tick_params(length = 5)
        ax.xaxis.set_tick_params(length = 5)
        ax.grid(color = '0.75', axis = 'y') 
        ax.set_yticklabels(['{:,.1f}'.format(x) for x in ax.get_yticks()], fontsize = tick_fontsize)
        ax.set_ylabel('Loss ratio', fontsize = axis_label_fontsize)
        if include_legend == True: ax.legend(frameon = False, fontsize = legend_fontsize, loc='center left', bbox_to_anchor=(1, 0.5))
    fig = plt.gcf()  
    plot_as_emf(fig, filename = '%s_losses_at_imls.emf' % (frame_name))
    plt.close()
    
#--- Defining directories
global case
case = 1
option = 'HAZUS'
directory = Path.cwd()
results_path = directory
for file in os.listdir(results_path):
    if file.endswith('.pickle') and file.startswith('ida_'):
        print(file)
        ida_pickle = results_path/file
        pickle_file = open(ida_pickle, 'rb')
        results = pickle.load(pickle_file)
        pickle_file.close()
        response_spectra_path = directory/'RS.pickle'
        RS = pd.read_pickle(response_spectra_path)
        periods_path = directory/'Periods.tcl'
        periods = pd.read_csv(periods_path, index_col  = None)
        # Calculate PGA if not output via OpenSees
#        results = _calc_IDA_PGA(ida_pickle, response_spectra_path, periods_path)
        iml_max = 5; iml_step = 0.05; iml_range = np.arange(iml_step, iml_max + iml_step, iml_step)
#        plot_edps_at_imls(ida_pickle, results,response_spectra_path,periods_path)
#        plot_collapse_fragility(results,iml_range)
#        plot_demolition_fragility(results,iml_range)
        plot_loss_curves(ida_pickle,results,iml_range,response_spectra_path,periods_path,option)





# Function for timing
def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

# --------- Stop the clock and report the time taken in seconds
elapsed = timeit.default_timer() - start_time
print('Running time: ',truncate(elapsed,1), ' seconds')
print('Running time: ',truncate(elapsed/float(60),2), ' minutes')




