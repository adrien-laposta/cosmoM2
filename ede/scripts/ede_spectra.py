import matplotlib.pyplot as plt
import numpy as np
from classy import Class
import os
import sys
import matplotlib
plt.style.use('seaborn-whitegrid')

save_path = '../figures/power_spectra/'
if not os.path.isdir(save_path):
    os.makedirs(save_path)



lambda_cdm_params = { "output":'tCl,pCl,lCl',
                      "lensing":'yes',
                      "h":0.6821,
                      "omega_b":0.02253,
                      "omega_cdm":0.1177,
                      "A_s": 2.216e-09,
                      "n_s":0.9686,
                      "tau_reio":0.085,
                      "m_ncdm":0.06,
                      "N_ncdm":1,
                      "N_ur":2.0328,
                      "l_max_scalars":4500,
                      "non linear":'HMcode'}
lambda_cdm = Class()
lambda_cdm.set(lambda_cdm_params)
lambda_cdm.compute()

ede = Class()
ede.set({'h': .7219,
         'fEDE': 0.122,
         'log10z_c': 3.562,
         'thetai_scf': 2.83,
         'A_s': 2.215e-09,
         'n_s': 0.9889,
         'omega_b': 0.02253,
         'omega_cdm': 0.1306,
         'tau_reio': 0.072,
         'non linear':'HMCODE',
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
         'output':'tCl pCl lCl',
         'lensing':'yes',
         'l_max_scalars':4500})
ede.compute()

cl_ede = ede.lensed_cl()
cl_lcdm = lambda_cdm.lensed_cl()

ell = cl_lcdm.get("ell")

modes = ["tt", "ee", "te"]

for mode in modes:
    fig = plt.figure(figsize = (18,10))

    ax = fig.add_subplot(121)
    ax.grid(True,linestyle='dotted')
    d_ell_ede = 1e12*ell*(ell+1)/2/np.pi*cl_ede.get(mode)
    d_ell_lcdm = 1e12*ell*(ell+1)/2/np.pi*cl_lcdm.get(mode)
    d_ell_ede = d_ell_ede[2:]
    d_ell_lcdm = d_ell_lcdm[2:]
    ax.plot(ell[2:], d_ell_ede, label="EDE model", color = 'darkred', lw=3)
    ax.plot(ell[2:], d_ell_lcdm, label=r"$\Lambda$-CDM model", color = 'k', linestyle = 'dotted',lw=3)
    ax.set_yscale('log')
    ax.set_xlabel(r'$\ell$', fontsize = 18)
    ax.set_ylabel(r'$\mathcal{{D}}_{{\ell}}^{{{0}}}$'.format(mode.upper()), fontsize = 18)
    plt.legend()

    ax = fig.add_subplot(122)
    ax.grid(True, linestyle='dotted')
    ax.set_ylabel(r'$\frac{{ \mathcal{{D}}_{{\ell}}^{{{{{0}}}, EDE}} - \mathcal{{D}}_{{\ell}}^{{{{{0}}}, CDM}} }}{{ \mathcal{{D}}_{{\ell}}^{{{{{0}}}, CDM}} }}$'.format(mode.upper()), fontsize = 18)
    ax.set_xlabel(r'$\ell$', fontsize = 18)
    ax.set_ylim(-0.02, 0.02)
    ax.plot(ell[2:], (d_ell_ede - d_ell_lcdm) / d_ell_lcdm, color = 'k', lw = 3)
    plt.savefig(save_path + "{0}.png".format(mode), dpi=300)
