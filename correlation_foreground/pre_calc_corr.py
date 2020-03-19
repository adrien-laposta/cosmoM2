import os

import camb
import numpy as np

import mflike as mfl
from pspy import so_dict

# from toolbox import *


def noise_freq(freq_list, noise_data_path, ell_max):

    n_freqs = len(freq_list)
    noise_list_diag = []

    for i, freq in enumerate(freq_list):
        noise_t_spectrum = np.loadtxt(
            os.path.join(noise_data_path,
                         "noise_t_LAT_{0}xLAT_{0}.dat".format(freq)))[:, 1]
        noise_list_diag.append(noise_t_spectrum[:ell_max - 1])

    n_ell = ell_max - 1
    noise_list_matrix = []

    for i in range(n_ell):
        noise_matrix = np.zeros((n_freqs, n_freqs))
        for j in range(n_freqs):
            noise_matrix[j, j] = noise_list_diag[j][i]
        noise_list_matrix.append(noise_matrix)

    return np.array(noise_list_matrix)


def get_cl_array(cosmo_parameters, fg_parameters, ell_max, freq_list,
                 noise_list_matrix):

    d = so_dict.so_dict()
    d.read_from_file("global_healpix_example.dict")
    fg_norm = d["fg_norm"]
    components = {"tt": d["fg_components"], "ee": [], "te": []}
    fg_model = {"normalisation": fg_norm, "components": components}
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

    TT = totCL[:, 0][2:]
    ell_list = np.arange(len(TT) + 2)[2:]

    power_spectrum_matrix = []
    noise_list_matrix = [
        element[:len(freq_list), :len(freq_list)]
        for element in noise_list_matrix
    ]

    fg_dict = mfl.get_foreground_model(fg_params, fg_model, freq_list,
                                       ell_list)

    n_ell = len(ell_list)
    n_freqs = len(freq_list)

    for ell in range(n_ell):
        c_matrix = np.ones((n_freqs, n_freqs)) * TT[ell]
        for i in range(n_freqs):
            for j in range(n_freqs):
                c_matrix[i, j] = c_matrix[i, j] + noise_list_matrix[ell][
                    i, j] + fg_dict['tt', 'all', freq_list[i],
                                    freq_list[j]][ell]
        power_spectrum_matrix.append(c_matrix)

    return np.array(power_spectrum_matrix)


def get_cl_derivatives(cosmo_parameters, fg_parameters, ell_max, freq_list,
                       noise_list_matrix):

    n_params_cosmo = len(cosmo_parameters)
    n_params_fg = len(fg_parameters)
    epsilon_cosmo = np.array(cosmo_parameters) / 100
    epsilon_fg = np.array(fg_parameters) / 100
    var_temp = []

    for i in range(n_params_cosmo):
        eps = epsilon_cosmo[i] * np.eye(1, n_params_cosmo, i)
        eps = eps.flatten()
        CL_plus = get_cl_array(cosmo_parameters + eps, fg_parameters, ell_max,
                               freq_list, noise_list_matrix)
        CL_moins = get_cl_array(cosmo_parameters - eps, fg_parameters, ell_max,
                                freq_list, noise_list_matrix)
        der = (CL_plus - CL_moins) / (2 * epsilon_cosmo[i])
        var_temp.append(der)

    for j in range(n_params_fg):
        eps = epsilon_fg[j] * np.eye(1, n_params_fg, j)
        eps = eps.flatten()
        CL_plus = get_cl_array(cosmo_parameters, fg_parameters + eps, ell_max,
                               freq_list, noise_list_matrix)
        CL_moins = get_cl_array(cosmo_parameters, fg_parameters - eps, ell_max,
                                freq_list, noise_list_matrix)
        der = (CL_plus - CL_moins) / (2 * epsilon_fg[j])
        var_temp.append(der)

    return np.array(var_temp)


def pre_calculation(cosmo_parameters, fg_parameters, ell_max, freq_list,
                    noise_data_path, save_path, names):

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    noise_list_matrix = noise_freq(freq_list, noise_data_path, ell_max)
    np.save(os.path.join(save_path, 'noise'), noise_list_matrix)

    power_spectrums = get_cl_array(cosmo_parameters, fg_parameters, ell_max,
                                   freq_list, noise_list_matrix)
    np.save(os.path.join(save_path, 'CL'), power_spectrums)

    deriv = get_cl_derivatives(cosmo_parameters, fg_parameters, ell_max,
                               freq_list, noise_list_matrix)
    for i, name in enumerate(names):
        np.save(os.path.join(save_path, 'deriv_' + name), deriv[i])
