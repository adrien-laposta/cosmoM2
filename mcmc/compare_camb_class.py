import matplotlib.pyplot as plt
import numpy as np
import toolbox
from getdist import MCSamples
import getdist.plots as gdplt
from getdist.mcsamples import loadMCSamples


fisher_sigma = np.loadtxt(
                '../true_fisher_forecast/saves/datas/sigmas_all_binned.dat'
                ) # fisher forecasted sigmas


true_params = np.array([
                       67.66, 0.02242, 0.11933, 3.047, 0.9665, 0.0561
                       ]) # parameters used to create the simulation of data

name_param = [r'$H_0$', r'$\Omega_bh^2$', r'$\Omega_ch^2$',
              r'$ln(10^{10}A_s)$', r'$n_s$', r'$\tau$'] # latex for parameters

mean_class = np.loadtxt('saves/data/class_param_new.dat')
mean_camb = np.loadtxt('saves/data/camb_param.dat')
sigma_class = np.loadtxt('saves/data/class_sigma_new.dat')
sigma_camb = np.loadtxt('saves/data/camb_sigma.dat')

path_camb = 'output_camb3/mcmc'
path_class = 'output_class3/mcmc'
wanted_params_camb = ['H0','ombh2','omch2','logA','ns','tau']
wanted_params_class = ['H0','omega_b','omega_cdm','logA','n_s','tau_reio']

gd_sample_camb = loadMCSamples(path_camb, settings = {'ignore_rows': 0.8})
gd_sample_class = loadMCSamples(path_class, settings = {'ignore_rows': 0.25})
gdplot = gdplt.getSubplotPlotter()

fig = plt.figure(figsize=(18, 10))
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

for i, name in enumerate(name_param):
    ax = fig.add_subplot(231 + i)
    ax.grid(True, linestyle = '--')

    x = np.linspace(true_params[i] - 5 * fisher_sigma[i],
                    true_params[i] + 5 * fisher_sigma[i], 500)

    y_fish = toolbox.gaussian(x, true_params[i], fisher_sigma[i])
    #y_class = toolbox.gaussian(x, mean_class[i], sigma_class[i])
    #y_camb = toolbox.gaussian(x, mean_camb[i], sigma_camb[i])

    #ax.plot(x, y_fish / np.max(y_fish), label = 'Fisher forecast',
            #color = 'darkblue', lw = 2)
    #ax.plot(x, y_class / np.max(y_class), label = 'class MCMC forecast', color = 'darkred')
    #ax.plot(x, y_camb / np.max(y_camb), label = 'camb MCMC forecast', color = 'darkgreen')
    #ax.axvline(best_fit[i], linestyle = '--', color = 'k', label = 'Best fit')
    gdplot.add_1d(gd_sample_camb, wanted_params_camb[i], ax = ax,
                  normalized = False, color = 'darkred',
                  label = 'MCMC forecast camb', lw=2)
    gdplot.add_1d(gd_sample_class, wanted_params_class[i], ax = ax,
                  normalized = False, color = 'darkgreen',
                  label = 'MCMC forecast class', lw=2)
    ax.set_ylim(0,1)
    #ax.set_xlim(true_params[i] - 5 * fisher_sigma[i],
                #true_params[i] + 5 * fisher_sigma[i])
    #ax.fill_between(x, -0.5, 1.5, where = np.abs(x-true_params[i])<=fisher_sigma[i],
    #facecolor='k', alpha=0.3)
    #ax.fill_between(x, -0.5, 1.5, where = np.abs(x-true_params[i])<=2*fisher_sigma[i],
    #facecolor='darkgray', alpha=0.3)
    ax.set_title(name, fontsize=16)

    handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc = 'upper right', fontsize=14)
plt.tight_layout()
fig.savefig("saves/figures/forecast_mcmc_compared_v2.png", dpi = 300)
