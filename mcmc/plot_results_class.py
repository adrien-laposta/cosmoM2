from getdist.mcsamples import loadMCSamples
import os
import matplotlib.pyplot as plt
import getdist.plots as gdplt
import numpy as np
from getdist import MCSamples
import sys
import toolbox

######################################################
#================== Initialisation ==================#
######################################################

path = 'output_class3/mcmc'
save_path = 'saves/figures/class/'
save_path_dat = 'saves/data/'

wanted_params = ['H0','omega_b','omega_cdm','logA','n_s','tau_reio'] # parameters to plot

index = [7, 3, 4, 0, 1, 5] # index of the parameters in the txt file

fisher_sigma = np.loadtxt(
                '../true_fisher_forecast/saves/datas/sigmas_all_binned.dat'
                ) # fisher forecasted sigmas


true_params = np.array([
                       67.66, 0.02242, 0.11933, 3.047, 0.9665, 0.0561
                       ]) # parameters used to create the simulation of data

name_param = [r'$H_0$', r'$\Omega_bh^2$', r'$\Omega_ch^2$',
              r'$ln(10^{10}A_s)$', r'$n_s$', r'$\tau$'] # latex for parameters

delimiter = np.array([1, 16, 32, 48, 64, 80, 96, 112, 128,
                      144, 160, 176, 193, 209, 227
                      ]) # where ends each column of the txt file

column_loc = delimiter - 1
widths = column_loc[1:] - column_loc[:-1]


names = ['weight', 'minuslogpost', 'logA', 'n_s', 'theta_s_1e2', 'omega_b',
         'omega_cdm', 'tau_reio', 'A_s', 'H0', 'minuslogprior',
         'minuslogprior__0', 'chi2', 'chi2__so_forecast'
         ] # names of each column

######################################################

######################################################
#================== Triangle plot  ==================#
######################################################

gd_sample = loadMCSamples(path, settings = {'ignore_rows': 0.3})
gdplot = gdplt.getSubplotPlotter()
gdplot.triangle_plot(gd_sample, wanted_params, filled = True)
plt.savefig(save_path+"constraints.png", dpi = 300)

######################################################

###########################################################
#================== Mean values / Stds  ==================#
###########################################################

mean = gd_sample.getMeans()[:-4] # remove chi2 means
covmat = gd_sample.getCov()[:-4,:-4]
sigma = np.sqrt(np.diag(covmat)) # standard deviations

mcmc_params = []
mcmc_sigma = []

for element in index:
    mcmc_params.append(mean[element]) # take only the mean
    mcmc_sigma.append(sigma[element]) # of wanted parameters

mcmc_params = np.array(mcmc_params)
mcmc_sigma = np.array(mcmc_sigma)
np.savetxt(save_path_dat+'class_param_new.dat', mcmc_params)
np.savetxt(save_path_dat+'class_sigma_new.dat', mcmc_sigma)

###########################################################

############################################################
#================== Chains for each job  ==================#
############################################################

for parameter in (np.append(wanted_params,'chi2')):
    list_dict = []
    variable = parameter

    fig = plt.figure(figsize = (12, 12))

    for i in range(1,5):
        ax = fig.add_subplot(220 + i)
        ax.grid(True, linestyle = 'dotted')
        file = path +".{0}.txt".format(i)
        data = np.genfromtxt(file, dtype = None, delimiter = widths, autostrip = True)
        nrow = len(data)
        ncol = len(data[0])
        chains = {}
        for element in names:
            chains[element] = []
        for a in range(nrow):
            for j in range(ncol):
                chains[names[j]].append(data[a][j])
        list_dict.append(chains)
        ax.plot(chains[variable], color = "darkred")
        ax.set_title('Job {0}'.format(i))
        fig.suptitle('{0}'.format(variable))
        plt.savefig(save_path+"{0}_chains.png".format(variable), dpi = 300)

list_index_chi2min = []
for element in list_dict:
    list_index_chi2min.append([np.argmin(element['chi2']), np.min(element['chi2'])])
list_index_chi2min = np.array(list_index_chi2min)
min_job = int(np.argmin(list_index_chi2min[:, 1]))
min_row = int(list_index_chi2min[min_job, 0])
best_fit = []
for element in index:
    best_fit.append(list_dict[min_job][names[element+2]][min_row])


############################################################

#######################################################
#================== Fisher vs MCMC  ==================#
#######################################################

fig = plt.figure(figsize=(18, 10))
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

for i, name in enumerate(wanted_params):
    ax = fig.add_subplot(231 + i)
    ax.grid(True, linestyle = '--')

    x = np.linspace(true_params[i] - 5 * fisher_sigma[i],
                    true_params[i] + 5 * fisher_sigma[i], 500)

    y_fish = toolbox.gaussian(x, true_params[i], fisher_sigma[i])
    #y_mcmc = toolbox.gaussian(x, mcmc_params[i], mcmc_sigma[i])
    ax.plot(x, y_fish/np.max(y_fish), label = 'Fisher forecast',
            color = 'darkblue')
    #ax.plot(x, y_mcmc / np.max(y_mcmc), label = 'MCMC forecast class', color = 'darkred')
    gdplot.add_1d(gd_sample, name, ax = ax, normalized = False, color = 'darkred', label = 'MCMC forecast class', lw=2)
    ax.set_xlim(true_params[i] - 5 * fisher_sigma[i], true_params[i] + 5 * fisher_sigma[i])
    #for j in range(0,4):
        #mean_mcmc = np.mean(list_dict[0][name][int(len(list_dict[j][name])/2):])
        #sigma_mcmc = np.std(list_dict[0][name][int(len(list_dict[j][name])/2):])
        #y_mcmc = toolbox.gaussian(x, mean_mcmc, sigma_mcmc)
        #ax.plot(x, y_mcmc/np.max(y_mcmc), color='k', label ='MCMC forecast class')
    #ax.axvline(best_fit[i], linestyle = '--', color = 'k', label = 'Best fit')
    ax.set_ylim(0,1)
    ax.fill_between(x, -0.5, 1.5, where = np.abs(x-true_params[i])<=fisher_sigma[i],
    facecolor='k', alpha=0.3)
    ax.fill_between(x, -0.5, 1.5, where = np.abs(x-true_params[i])<=2*fisher_sigma[i],
    facecolor='darkgray', alpha=0.3)
    ax.set_title(name_param[i], fontsize=16)

    handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc = 'upper right', fontsize=14)
plt.tight_layout()
fig.savefig(save_path+"forecast_mcmc.png", dpi = 300)

#######################################################
