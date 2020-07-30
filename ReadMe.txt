Functions

plot_as_emf - Saving figures as an emf
_calc_repair_cost - Calculation of repair costs
_calc_SaT1_475_mean - Calculation of mean SaT1 of 475 years
_calc_IDA_PGA - Calculation of PGA associated with each IML of record
_spline - Doing a spline for edps and imls
_mlefit - Fitting the lognormal CDF
_calc_gama - Calculation of gama, SaT1 modifier for different RPs
_calc_collapse_fragility - Calculation of collapse fragility
_calc_p_edp_given_im - Calculation of EDPvs PoE fragility for a given IML in terms of PDF
_calc_demolition_fragility - Calculation of demolition fragility
_calc_losses - Calculation of losses
plot_component_loss_ratios_HAZUS - Plotting component loss ratios (HAZUS)
plot_demolition_ls - Plotting demolition losses
plot_edps_at_imls - Plotting edps at imls
plot_collapse_fragility - Plotting collapse fragility
plot_demolition_fragility - Plotting demolition fragility
plot_loss_curves - Plotting loss curves


Files necessary:
* ida_pickle - dictionary that includes IDA results
	* IDA
		* Records
			* IML, ISDR, PFA, RISDR
	* summary_results
		* Records
			* IML
				* maxFA, maxISDR, maxRISDR
* Periods - dataframe with 
	* columns as Typology, and 
	* rows as cases
	- or if 1 case is being considered, supply just the period
		- do a check with the length of an array
Ready:
* Response Spectra - datafrane with 
	* first columns of T1
	* other columns as Record names
	* Rows are untagged


Sequence of calculations
0. Defines inputs - RS, periods, ida_pickle-> results
1. _calc_IDA_PGA - updates results to include PGA in maxPFA in 'summary_results'
2. plot_edps_at_imls 
	2a. _calc_SaT1_475_mean
	2b. _calc_gama
3. plot_collapse_fragility
	3a. _calc_collapse_fragility
4. plot_demolition_fragility
	4a. _calc_demolition_fragility
5. plot_loss_curves
	5a. _calc_losses


MODIFY THOSE THINGS ONCE THE COLLAPSE SAFETY STUFF IS FINALIZED
Update pathes 
Collapse fragility not showing properly - to be verified at a given ductility, essentially flatten at that point
Add hazard function
Run for case 1 to verify that the results make at least some sense 
