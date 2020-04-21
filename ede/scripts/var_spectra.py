import matplotlib.pyplot as plt
import numpy as np
from classy import Class
import os
import sys
import matplotlib as mpl
plt.style.use('seaborn-whitegrid')

save_path = '../igures/power_spectra_variations/'
if not os.path.isdir(save_path):
    os.makedirs(save_path)

def get_spectra(fEDE, log10z_c, thetai_scf):

    ede = Class()
    ede.set({'100*theta_s': 1.04119,    #0.7219
             'fEDE': fEDE, #0.122
             'log10z_c': log10z_c, #3.562
             'thetai_scf': thetai_scf, #2.83
             'A_s': 2.215e-09,
             'n_s': 0.9889,
             'omega_b': 0.02253,
             'omega_cdm': 0.1306,
             'tau_reio': 0.072,
             'non linear':'halofit',
             'N_ncdm':1,
             'm_ncdm': 0.06,
             'N_ur':2.0328,
             'Omega_Lambda':0.0,
             'Omega_fld':0,
             'Omega_scf':-1,
             'n_scf':3,
             'CC_scf':1,
             'scf_parameters':'1, 1, 1, 1, 1, 0.0',
             'scf_tuning_index':3,
             'attractor_ic_scf':'no',
             'output':'tCl pCl lCl mPk',
             'lensing':'yes',
             'l_max_scalars':3508,
             'P_k_max_h/Mpc': 100,
             'z_max_pk': 3.})

    ede.compute()

    cl_ede = ede.lensed_cl()
    ell = cl_ede.get("ell")[2:]
    modes = ["tt", "ee", "te"]
    dict_d_ell = {}
    for mode in modes:
        dict_d_ell[mode] = 1e12*ell*(ell+1)/2/np.pi*cl_ede.get(mode)[2:]
    dict_d_ell["ell"] = ell
    return(dict_d_ell)

def var_spectra(mean_parameters, index_param_var, Npts):

    if index_param_var == 0:
        range_param = np.linspace(0.001,
                                  0.30,
                                  Npts)
    elif index_param_var == 2:
        range_param = np.linspace(0.5 * mean_parameters[index_param_var],
                                  1.1 * mean_parameters[index_param_var],
                                  Npts)
    elif index_param_var == 1:
        range_log = np.linspace(0.5 * pow(10, mean_parameters[index_param_var]),
                                1.5 * pow(10, mean_parameters[index_param_var]),
                                Npts)
        range_param = np.log(range_log) / np.log(10)
    cl_tt = []
    cl_ee = []
    cl_te = []
    r_te = []
    classic_param = mean_parameters.copy()
    for param in range_param:
        print(param)
        classic_param[index_param_var] = param
        dict_c = get_spectra(classic_param[0],
                           classic_param[1],
                           classic_param[2])
        cl_tt.append(dict_c['tt'])
        cl_ee.append(dict_c['ee'])
        cl_te.append(dict_c['te'])
        r_te.append(dict_c['te'] / np.sqrt(dict_c['ee'] * dict_c['tt']))
    cl_tt = np.array(cl_tt)
    cl_ee = np.array(cl_ee)
    cl_te = np.array(cl_te)
    r_te = np.array(r_te)

    var_cl_dict = {}
    var_cl_dict['tt'] = cl_tt
    var_cl_dict['ee'] = cl_ee
    var_cl_dict['te'] = cl_te
    var_cl_dict['r'] = r_te
    ell = dict_c['ell']
    var_cl_dict['ell'] = ell

    return(var_cl_dict)

#==============================================================================#

parameters = [0.122, 3.562, 2.83]
Npts = 10

#latex_params = [r'$log_{10}(z_c)$', r'$\theta_i$', r'$f_{EDE}$',]
latex_params = [r'$f_{EDE}$', r'$z_c$', r'$\theta_i$']

#names = ['log10z_c', 'theta_i', 'fEDE']
names = ['fEDE', 'log10z_c', 'theta_i']

index_str = sys.argv[1]
if index_str == 'fEDE':
    index = 0
elif index_str == 'log10z_c':
    index = 1
elif index_str == 'theta_i':
    index = 2


if index == 1:
    c = np.linspace(-50, 50, Npts)
elif index == 2:
    c = np.linspace(-50, 10, Npts)
elif index == 0:
    c = np.linspace(0, 0.30, Npts)
norm = mpl.colors.Normalize(vmin = c.min(), vmax = c.max())
cmap = mpl.cm.ScalarMappable(norm = norm, cmap = mpl.cm.jet)
cmap.set_array([])


print("-"*10)
print("dict import")
print("-"*10)
var_dict = var_spectra(parameters, index, Npts)
print("import ended")
print("-"*10)
for key in var_dict:
    if key != 'ell':
        y = var_dict[key]
        x = var_dict['ell']
        fig, ax = plt.subplots(figsize = (16, 9))
        for j, yi in enumerate(y):
            ax.plot(x, yi, c = cmap.to_rgba(c[j]))
        ax.set_xlabel(r'$\ell$', fontsize = 18)
        if key == 'r':
            ax.set_ylabel(r'$\mathcal{R}_{\ell}^{TE}$', fontsize = 18)
        else:
            ax.set_ylabel(r'$\mathcal{{D}}_{{\ell}}^{{{0}}}$'.format(key.upper()), fontsize = 18)
        ax.grid(True, linestyle = 'dotted')
        if index == 2:
            cbar = fig.colorbar(cmap, ticks = [-50, 0 , 10])
        elif index == 1:
            cbar = fig.colorbar(cmap, ticks = [-50, 0 , 50])
        elif index == 0:
            cbar = fig.colorbar(cmap, ticks = [0, 0.1, 0.2, 0.3])
        if index == 2:
            cbar.ax.set_yticklabels([r'$-50\%$', r'$\theta_i = {{{0}}}$'.format(parameters[2]), r'$+10\%$'])
        elif index == 1:
            cbar.ax.set_yticklabels([r'$-50\%$', r'$z_c = {{{0}}}$'.format(round(10 ** parameters[1], 1)), r'$+50\%$'])
        elif index == 0:
            #cbar.ax.set_yticklabels([r'$-50\%$', r'$f_{{EDE}} = {{{0}}}$'.format(parameters[0]), r'$+50\%$'])
            cbar.set_label(latex_params[index], fontsize = 18)
        fig.savefig(save_path + '{0}_{1}'.format(key, names[index]))
