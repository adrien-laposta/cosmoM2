import matplotlib.pyplot as plt
import os
import numpy as np
import toolbox

cosmo_parameters = [67.4, 0.02207, 0.1196, 3.098, 0.9616, 0.097]
fg_parameters = [3.3, 1.66, 6.91, 2.07, 4.88, 2.2, 3.09]

name_param_fig = [
    r'$H_0$', r'$\Omega_bh^2$', r'$\Omega_ch^2$', r'$ln(10^{10}A_s)$',
    r'$n_s$', r'$\tau$', r'$a_{tSZ}$', r'$a_{kSZ}$', r'$a_p$', r'$\beta_p$',
    r'$a_c$', r'$\beta_c$', r'$a_s$'
]

save_path_fig = 'saves/figures/'
save_path_dat = 'saves/datas/'



fig = plt.figure(figsize=(24, 13.5))
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
colors = dict()
colors['binned'] = 'darkblue'
colors['unbinned'] = 'darkred'
sigmas = dict()

sigmas_temp_binned = np.loadtxt(
                     save_path_dat + "sigmas_all_binned.dat")
sigmas_temp_unbinned = np.loadtxt(
                      save_path_dat + "sigmas_all.dat")
sigmas['binned'] = sigmas_temp_binned
sigmas['unbinned'] = sigmas_temp_unbinned

for i, cosmo_parameter in enumerate(cosmo_parameters):
    ax = fig.add_subplot(231 + i)
    ax.grid(True, linestyle='--')

    x = np.linspace(cosmo_parameter - 4 * sigmas['binned'][i],
                    cosmo_parameter + 4 * sigmas['binned'][i], 500)
    for key in sigmas:
        sigma = sigmas[key][i]
        y = toolbox.gaussian(x, cosmo_parameter, sigma)

        ax.plot(x, y / np.max(y), label = key,
                color = colors[key], linewidth = 2.5)

    ax.set_title(name_param_fig[i], fontsize=20)
    handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', fontsize=18)
fig.tight_layout()

fig.savefig(os.path.join(save_path_fig, "forecast_binned_unbinned.png"), dpi=300)
