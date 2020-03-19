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
