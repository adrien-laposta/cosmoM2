import os
import matplotlib.pyplot as plt
import numpy as np
import toolbox
import pickle
import mflike as mfl
from pspy import so_dict
import camb

noise_data_path = 'sim_data/noise_tot_test/'
save_path_fig = 'saves/figures/'
save_path_dat = 'saves/datas/'
cl_path = 'pre_calc/'

ell_max = 4500
n_split = 2
bin_width = 20

planck_parameters = [67.66, 0.02242, 0.11933, 3.047, 0.9665, 0.0561]
fg_parameters = [3.3, 1.66, 6.91, 2.07, 4.88, 2.2, 3.09]


noise_planck_t = np.loadtxt(
                 noise_data_path+'noise_t_Planck_143xPlanck_143.dat'
                 )[:, 1][:ell_max -1]
noise_planck_pol = np.loadtxt(
                   noise_data_path + 'noise_pol_Planck_143xPlanck_143.dat'
                   )[: ,1][:ell_max -1]

noise_planck_t = toolbox.get_binned_list(noise_planck_t, bin_width)
noise_planck_pol = toolbox.get_binned_list(noise_planck_pol, bin_width)


def get_cl_dict(cosmo_parameters, fg_parameters, ell_max, bin_width):

    d = so_dict.so_dict()
    d.read_from_file('global_healpix_example.dict')
    fg_norm = d['fg_norm']
    components = {'tt': d['fg_components'], 'ee': [], 'te': []}
    fg_model = {'normalisation': fg_norm, 'components': components}
    fg_params = {
        'a_tSZ': fg_parameters[0],
        'a_kSZ': fg_parameters[1],
        'a_p': fg_parameters[2],
        'beta_p': fg_parameters[3],
        'a_c': fg_parameters[4],
        'beta_c': fg_parameters[5],
        'n_CIBC': 1.2,
        'a_s': fg_parameters[6],
        'T_d': 9.6
    }

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=cosmo_parameters[0],
                       ombh2=cosmo_parameters[1],
                       omch2=cosmo_parameters[2],
                       mnu=0.06,
                       omk=0,
                       tau=cosmo_parameters[5])

    pars.InitPower.set_params(As=1e-10 * np.exp(cosmo_parameters[3]),
                              ns=cosmo_parameters[4],
                              r=0)

    pars.set_for_lmax(ell_max - 1, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
    totCL = powers['total']

    EE = totCL[:, 1][2:]
    TT = totCL[:, 0][2:]
    TE = totCL[:, 3][2:]

    ell_list = np.arange(len(TT) + 2)[2:]


    fg_dict = mfl.get_foreground_model(fg_params, fg_model, [143],
                                       ell_list)


    c_ell_dict = dict()

    c_ell_dict['tt', '143x143'] = toolbox.get_binned_list(
                                  TT + fg_dict['tt', 'all', 143, 143],
                                  bin_width)

    c_ell_dict['ee', '143x143'] = toolbox.get_binned_list(
                                  EE + fg_dict['ee', 'all', 143, 143],
                                  bin_width)


    c_ell_dict['te', '143x143'] = toolbox.get_binned_list(
                                  TE + fg_dict['te', 'all', 143, 143],
                                  bin_width)

    return c_ell_dict

def get_covariance_matrix(cosmo_parameters, fg_parameters, ell_max,
                          noise_t, noise_pol, n_split, bin_width):


    power_spectrums = get_cl_dict(cosmo_parameters, fg_parameters,
                                  ell_max, bin_width)

    ell_list = toolbox.get_binned_list(np.arange(ell_max+1)[2:], bin_width)

    N = n_split*(n_split-1)
    covmat_dict = dict()

    pre_fact = 1 / (2 * ell_list + 1) / bin_width

    covmat_dict['tt','tt','143x143','143x143'] = pre_fact * (
    2 * power_spectrums['tt','143x143'] * power_spectrums['tt','143x143'] + (
    2 / N * noise_t * noise_t * n_split * n_split)
    + 4 * power_spectrums['tt','143x143'] * noise_t
    )

    covmat_dict['ee','ee','143x143','143x143'] = pre_fact * (
    2 * power_spectrums['ee','143x143'] * power_spectrums['ee','143x143'] + (
    2 / N * noise_pol * noise_pol * n_split * n_split)
    +  4 * power_spectrums['ee','143x143'] * noise_pol
    )

    covmat_dict['te','te','143x143','143x143'] = pre_fact * (
        power_spectrums['tt','143x143'] * power_spectrums['ee','143x143'] + (
        power_spectrums['te','143x143'] * power_spectrums['te','143x143']
        ) + 1 / n_split * (power_spectrums['tt','143x143'
        ] * noise_pol + power_spectrums['ee','143x143'] * noise_t) + 1 / N * (
        noise_t * noise_pol * n_split * n_split
        ))

    return(covmat_dict)




###############################################################################

C_ell_binned_dict_so = pickle.load(
                    open(os.path.join(
                    cl_path, 'power_spectrums_binned.p'), 'rb'))

C_ell_binned_dict_planck = get_cl_dict(planck_parameters, fg_parameters,
                                       ell_max, bin_width)

covarmat_binned_so = pickle.load(
                     open(os.path.join(
                     cl_path, 'covariance_binned.p'), 'rb'))

covarmat_binned_planck = get_covariance_matrix(planck_parameters, fg_parameters,
                                               ell_max, noise_planck_t,
                                               noise_planck_pol, n_split,
                                               bin_width)

ell_list = toolbox.get_binned_list(np.arange(ell_max+1)[2:], bin_width)


for key in C_ell_binned_dict_planck:
    mode = key[0]
    C_planck = C_ell_binned_dict_planck[mode, '143x143']
    C_so = C_ell_binned_dict_so[mode, '145x145']

    sig_planck = np.sqrt(covarmat_binned_planck[mode,
                                                mode, '143x143',
                                                '143x143'])

    sig_so = np.sqrt(covarmat_binned_so[mode, mode, '145x145', '145x145'])
    so_max = 1.02 * np.max(C_so + sig_so)
    so_min = 1.02 * np.min(C_so - sig_so)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot()


    ax.plot(ell_list, C_planck, color = 'darkblue',
            label = "Planck Power Spectrum")

    ax.fill_between(ell_list, C_planck - sig_planck, C_planck + sig_planck,
                    color = 'darkblue', alpha = 0.2)

    ax.plot(ell_list, C_so, color = 'darkred', label = "SO Power Spectrum")
    ax.errorbar(ell_list, C_so, sig_so, fmt = '.',
                color = 'darkred', elinewidth = 1, capsize = 1.5)
    ax.set_ylim(so_min, so_max)
    ax.grid(True, linestyle = 'dotted',zorder = 20)
    ax.set_xlabel(r'$\ell$', fontsize = 20)
    ax.set_ylabel(r'$\mathcal{{D}}_{{\ell}}^{{{0}}}$'.format(mode),
                  fontsize = 20)
    #fig.suptitle('{} Power spectrum'.format(mode))
    fig.legend(loc='upper right', fontsize = 16)
    fig.tight_layout()
    fig.savefig(save_path_fig + '{0}_power_spectrum.png'.format(mode),
                dpi = 300)
