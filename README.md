# M2 Internship

### Correlation matrix with foregrounds
  - A plot showing the evolution of the correlation matrix for the 6 cosmological parameters and 7
    foreground parameters could be done with : `launch_corrmat.py` in the correlation_foreground
    directory.

  Datas (Power spectrum, its derivatives and noises) are pre-generated with `pre_calc_corr.py` for a
  set of given parameters.  These parameters are given in a list in the `launch_corrmat.py` script.

  If you need to generate again the datas, you just have to set the `calculate_data` variable to `True`.

  Executing the script `launch_corrmat.py` will save some figures and the covariance matrix for each
  size of the frequency list in `/saves/figures`. It also saves the data in `/saves/Data`

  The covariance matrix for the 145 GHz frequency only is not relevant. This is why this script
  could also plot the different normalized Fisher matrix (by setting the `plot_fisher` variable to
  `True`).

  You can also have a plot showing the influence of adding frequencies on the constraints on the
  cosmological parameters (by setting `plot_cosmo_parameters` to `True`).


### Forecast of cosmo parameters with CMB Likelihood
  - Everything needed to do that can be found in the true_fisher_forecast directory.

  The first thing to do is to generate datas with `pre_calc_fisher_upgraded.py`. You can generate these datas by
  setting the `calculate_data` variable to `True` in the `launch_cmb_fisher_upgraded.py` script. This variable is
  set to `True` by default because datas have to be computed.
  If the `binned` variable is set to true the datas (power spectrum, covariance matrix) will be generated with bins.

  Then, the `calculate_fisher_matrix` variable is used to generate and save the standard deviations for the cosmological
  parameters for all the `modes` we want.

  To plot the forecast you just have to set `plot_forecast` variable to `true`.

  - A little script, `compare_bin_unbin.py` is used to compare cosmological parameters constraints
  with and without binning. You just have to run this script if you want to do that.

  - The `binned_spectrum.py` is used to plot the binned spectrum and its errorbars for
  LAT145 and Planck143. You juste have to run the script.

  - To launch these two scripts, you have to compute and save datas first.
