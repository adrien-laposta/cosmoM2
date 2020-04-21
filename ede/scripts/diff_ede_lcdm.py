from classy import Class
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import toolbox

plt.style.use('seaborn-whitegrid')

save_path = '../figures/ede_cdm_comp/'
if not os.path.isdir(save_path):
    os.makedirs(save_path)

LCDM_pars = {'output': 'tCl,pCl,lCl',
             'lensing': 'yes',
             'h': 0.6744,
             'omega_b': 0.02244,
             'omega_cdm': 0.1201,
             'A_s': (1e-10)*np.exp(3.055),
             'n_s': 0.9659,
             'tau_reio': 0.0587,
             'm_ncdm': 0.06,
             'N_ncdm': 1,
             'N_ur': 2.0328,
             'l_max_scalars': 4500,
             'non linear': 'HMcode'}

EDE_pars = {'h': 0.6913,
            'omega_b': 0.02250,
            'omega_cdm': 0.1268,
            'A_s': (1e-10)*np.exp(3.056),
            'n_s': 0.9769,
            'tau_reio': 0.0539,
            'fEDE': 0.068,
            'log10z_c': 3.75,
            'thetai_scf': 2.96,
            'non linear': 'HMCODE',
            'N_ncdm': 1,
            'm_ncdm': 0.06,
            'N_ur': 2.0328,
            'Omega_Lambda': 0.0,
            'Omega_fld': 0,
            'Omega_scf': -1,
            'n_scf': 3,
            'CC_scf': 1,
            'scf_parameters': '1, 1, 1, 1, 1, 0.0',
            'scf_tuning_index': 3,
            'attractor_ic_scf': 'no',
            'output': 'tCl pCl lCl',
            'lensing': 'yes',
            'l_max_scalars': 4500}

LCDM = Class()
LCDM.set(LCDM_pars)
LCDM.compute()

EDE = Class()
EDE.set(EDE_pars)
EDE.compute()

bin_width = 30
c_ell_ede = EDE.lensed_cl()
c_ell_lcdm = LCDM.lensed_cl()

ell = c_ell_lcdm.get('ell')[2:]


modes = ['tt', 'ee', 'te', 'r']

dict_lcdm = {}
dict_ede = {}
dict_lcdm['tt'] = toolbox.get_binned_list(
                    1e12 * pow(2.7255, 2) * ell * (ell + 1) / 2 / np.pi * c_ell_lcdm.get('tt')[2:], bin_width)
dict_lcdm['ee'] = toolbox.get_binned_list(
                    1e12 * pow(2.7255, 2) * ell * (ell + 1) / 2 / np.pi * c_ell_lcdm.get('ee')[2:], bin_width)
dict_lcdm['te'] = toolbox.get_binned_list(
                    1e12 * pow(2.7255, 2) * ell * (ell + 1) / 2 / np.pi * c_ell_lcdm.get('te')[2:], bin_width)
dict_lcdm['r'] = toolbox.get_binned_list(
                    c_ell_lcdm.get('te')[2:] / np.sqrt(c_ell_lcdm.get('tt')[2:] * c_ell_lcdm.get('ee')[2:]),
                    bin_width)

dict_ede['tt'] = toolbox.get_binned_list(
                    1e12 * pow(2.7255, 2) * ell * (ell + 1) / 2 / np.pi * c_ell_ede.get('tt')[2:], bin_width)
dict_ede['ee'] = toolbox.get_binned_list(
                    1e12 * pow(2.7255, 2) * ell * (ell + 1) / 2 / np.pi * c_ell_ede.get('ee')[2:], bin_width)
dict_ede['te'] = toolbox.get_binned_list(
                    1e12 * pow(2.7255, 2) * ell * (ell + 1) / 2 / np.pi * c_ell_ede.get('te')[2:], bin_width)
dict_ede['r'] = toolbox.get_binned_list(
                    c_ell_ede.get('te')[2:] / np.sqrt(c_ell_ede.get('tt')[2:] * c_ell_ede.get('ee')[2:]),
                    bin_width)

ell = toolbox.get_binned_list(ell, bin_width)

cov_mat_binned_so = pickle.load(open(
                    '../../true_fisher_forecast/pre_calc/covariance_binned_30.p', 'rb'
                    ))

## choice of the 145 Ghz data

covariances = {}
covariances['tt'] = cov_mat_binned_so['tt', 'tt', '145x145', '145x145']
covariances['ee'] = cov_mat_binned_so['ee', 'ee', '145x145', '145x145']
covariances['te'] = cov_mat_binned_so['te', 'te', '145x145', '145x145']
covariances['r'] = (dict_lcdm['r']**2) * (
    covariances['te'] / pow(dict_lcdm['te'], 2) + 0.25 * (
    covariances['tt'] / pow(dict_lcdm['tt'], 2) + (
    covariances['ee'] / pow(dict_lcdm['ee'], 2))) - (
    (cov_mat_binned_so['tt','te','145x145','145x145'] / (dict_lcdm['tt'] * dict_lcdm['te'])) + (
    cov_mat_binned_so['ee', 'te', '145x145', '145x145'] / (dict_lcdm['ee'] * dict_lcdm['te'])
    )) + 0.5 * (cov_mat_binned_so['tt', 'ee', '145x145', '145x145'] / (
    dict_lcdm['tt'] * dict_lcdm['ee']
    )))

sigmas = {}
sigmas['tt'] = np.sqrt(covariances['tt'])
sigmas['ee'] = np.sqrt(covariances['ee'])
sigmas['te'] = np.sqrt(covariances['te'])
sigmas['r'] = np.sqrt(covariances['r'])

bmin = 1
bmax = 120

ell = ell[bmin:bmax]
for key in sigmas:
    sigmas[key] = sigmas[key][bmin:bmax]
    dict_lcdm[key] = dict_lcdm[key][bmin:bmax]
    dict_ede[key] = dict_ede[key][bmin:bmax]

for mode in modes:
    fig = plt.figure(figsize = (15,22))

    ax = fig.add_subplot(211)
    ax.grid(True,linestyle='dotted')
    if mode != 'tt':
        ax.plot(ell, dict_ede[mode], label = r"EDE model", color = 'darkorange', lw = 1)
        ax.plot(ell, dict_lcdm[mode], label = r'$\Lambda$-CDM model', color = 'k', lw = 1, linestyle = '--')
        ax.errorbar(ell, dict_lcdm[mode], sigmas[mode], fmt = '.',
                    color = 'k', elinewidth = 1, capsize = 1.5)
        so_max = 1.02 * np.max(dict_lcdm[mode] + sigmas[mode])
        so_min = 1.02 * np.min(dict_lcdm[mode] - sigmas[mode])
    else:
        ax.plot(ell, (ell**2) * dict_ede[mode], label = r"EDE model", color = 'darkorange', lw = 1)
        ax.plot(ell, (ell**2) * dict_lcdm[mode], label = r'$\Lambda$-CDM model', color = 'k', lw = 1, linestyle = '--')
        ax.errorbar(ell, (ell**2) * dict_lcdm[mode], (ell**2) * sigmas[mode], fmt = '.',
                    color = 'k', elinewidth = 1, capsize = 1.5)
        so_max = 1.02 * np.max((ell**2) * dict_lcdm[mode] + (ell**2) * sigmas[mode])
        so_min = 1.02 * np.min((ell**2) * dict_lcdm[mode] - (ell**2) * sigmas[mode])

    ax.set_ylim(so_min, so_max)
    ax.set_xlabel(r'$\ell$', fontsize = 18)
    if mode == 'r':
        ax.set_ylabel(r'$\mathcal{R}_{\ell}^{TE}$', fontsize = 18)
    elif mode == 'tt':
        ax.set_ylabel(r'$\ell^2\mathcal{{D}}_{{\ell}}^{{{0}}}$'.format(mode.upper()), fontsize = 18)
    else:
        ax.set_ylabel(r'$\mathcal{{D}}_{{\ell}}^{{{0}}}$'.format(mode.upper()), fontsize = 18)
    plt.legend()

    ax = fig.add_subplot(212)
    ax.grid(True, linestyle='dotted')
    if mode == 'r':
        ax.set_ylabel(r'$\frac{\left|\mathcal{R}_{\ell}^{{TE, EDE}} - \mathcal{R}_{\ell}^{{TE, CDM}}\right|}{\sigma(\mathcal{R}_{\ell}^{TE})^{CDM}}$', fontsize = 18)
    else:
        ax.set_ylabel(r'$\frac{{\left|\mathcal{{D}}_{{\ell}}^{{{{{0}}}, EDE}} - \mathcal{{D}}_{{\ell}}^{{{{{0}}}, CDM}}\right|}}{{ \sigma_{{\ell}}^{{{{{0}}}, CDM}} }}$'.format(mode.upper()), fontsize = 18)
    ax.set_xlabel(r'$\ell$', fontsize = 18)
    ax.plot(ell, np.abs(dict_ede[mode] - dict_lcdm[mode]) / sigmas[mode], color = 'k', lw = 3)
    plt.savefig(save_path + "{0}.png".format(mode), dpi=300)
    plt.savefig(save_path + "{0}.pdf".format(mode), dpi=300)
