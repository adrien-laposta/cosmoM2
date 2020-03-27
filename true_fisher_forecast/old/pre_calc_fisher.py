import os

import camb
import numpy as np

import mflike as mfl
from pspy import so_dict

# from toolbox import *


def get_noise_dict(freq_list, noise_data_path, ell_max):

    noise_dict = dict()

    for f1 in freq_list:
        for f2 in freq_list:

            if f1 == f2:
                noise_t_spectrum = np.loadtxt(
                    os.path.join(noise_data_path,
                                'noise_t_LAT_{0}xLAT_{0}.dat'.format(f1)))[:, 1]

                noise_dict['{0}x{1}'.format(f1, f2)] = noise_t_spectrum[
                                                                :ell_max - 1]

            else:
                noise_dict['{0}x{1}'.format(f1,f2)] = np.zeros(ell_max-1)

    return(noise_dict)



def get_cl_dict(cosmo_parameters, fg_parameters, ell_max, freq_list):

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

    TT = totCL[:, 0][2:]
    ell_list = np.arange(len(TT) + 2)[2:]


    fg_dict = mfl.get_foreground_model(fg_params, fg_model, freq_list,
                                       ell_list)

    n_freqs = len(freq_list)

    c_ell_dict = dict()

    for f1 in freq_list:
        for f2 in freq_list:
            if f2 <= f1:
                c_ell_dict['{0}x{1}'.format(f2,f1)] = TT + fg_dict['tt', 'all',
                                                                   f1, f2]

    return c_ell_dict



def get_cl_data_vector(cosmo_parameters, fg_parameters, ell_max, freq_list):

    cl_dict = get_cl_dict(cosmo_parameters, fg_parameters, ell_max, freq_list)
    c_ell_vector = []

    for f1 in freq_list:
        for f2 in freq_list:
            if f2 <= f1:
                c_ell_vector.append(cl_dict['{0}x{1}'.format(f2,f1)])

    return np.array(c_ell_vector)



def get_cl_derivatives(cosmo_parameters, fg_parameters, ell_max, freq_list):

    n_params_cosmo = len(cosmo_parameters)
    n_params_fg = len(fg_parameters)

    epsilon_cosmo = np.array(cosmo_parameters) / 100
    epsilon_fg = np.array(fg_parameters) / 100
    var_temp = []

    for i in range(n_params_cosmo):
        eps = epsilon_cosmo[i] * np.eye(1, n_params_cosmo, i)
        eps = eps.flatten()
        CL_plus = get_cl_data_vector(cosmo_parameters + eps, fg_parameters, ell_max,
                                     freq_list)
        CL_moins = get_cl_data_vector(cosmo_parameters - eps, fg_parameters, ell_max,
                                      freq_list)
        der = (CL_plus - CL_moins) / (2 * epsilon_cosmo[i])
        var_temp.append(der)

    for j in range(n_params_fg):
        eps = epsilon_fg[j] * np.eye(1, n_params_fg, j)
        eps = eps.flatten()
        CL_plus = get_cl_data_vector(cosmo_parameters, fg_parameters + eps, ell_max,
                                     freq_list)
        CL_moins = get_cl_data_vector(cosmo_parameters, fg_parameters - eps, ell_max,
                                      freq_list)
        der = (CL_plus - CL_moins) / (2 * epsilon_fg[j])
        var_temp.append(der)

    return np.array(var_temp)




def get_covariance_matrix(cosmo_parameters, fg_parameters, ell_max, freq_list,
                      noise_data_path,n_split):


    power_spectrums = get_cl_dict(cosmo_parameters, fg_parameters,
                                         ell_max, freq_list)
    cov_shape = len(power_spectrums)
    noise_dict = get_noise_dict(freq_list, noise_data_path, ell_max)

    ell_list = np.arange(len(power_spectrums['27x27']) + 2)[2:]

    covmat_list = []
    N = n_split*(n_split-1)

    for a, ell in enumerate(ell_list):
        covmat = np.zeros((cov_shape, cov_shape))

        for i, key1 in enumerate(power_spectrums):
            for j, key2 in enumerate(power_spectrums):

                freqs1 = key1.split('x')
                freqs2 = key2.split('x')
                i_j = [int(element) for element in freqs1]   ##i,j
                k_l = [int(element) for element in freqs2]   ##k,l
                ##This step is important in order to keep the keys
                ## 'f1xf2' in an ascending order

                i_k = [i_j[0], k_l[0]]
                i_l = [i_j[0], k_l[1]]
                j_k = [i_j[1], k_l[0]]
                j_l = [i_j[1], k_l[1]]
                i_k.sort()
                i_l.sort()
                j_k.sort()
                j_l.sort()

                pre_fact = 1/(2*ell+1)
                covmat[i, j] = power_spectrums[
                               '{0}x{1}'.format(i_k[0], i_k[1])
                               ][a] * power_spectrums[
                               '{0}x{1}'.format(j_l[0], j_l[1])
                               ][a] + power_spectrums[
                               '{0}x{1}'.format(i_l[0], i_l[1])
                               ][a] * power_spectrums[
                               '{0}x{1}'.format(j_k[0], j_k[1])
                               ][a] + (power_spectrums[
                               '{0}x{1}'.format(i_k[0], i_k[1])
                               ][a] * noise_dict[
                               '{0}x{1}'.format(j_l[0], j_l[1])
                               ][a] + power_spectrums[
                               '{0}x{1}'.format(j_l[0], j_l[1])
                               ][a] * noise_dict[
                               '{0}x{1}'.format(i_k[0], i_k[1])
                               ][a]) + (power_spectrums[
                               '{0}x{1}'.format(i_l[0], i_l[1])
                               ][a] * noise_dict[
                               '{0}x{1}'.format(j_k[0], j_k[1])
                               ][a] + power_spectrums[
                               '{0}x{1}'.format(j_k[0], j_k[1])
                               ][a] * noise_dict[
                               '{0}x{1}'.format(i_l[0], i_l[1])
                               ][a]) + (1/N) * pow(n_split,2) * (
                               noise_dict[
                               '{0}x{1}'.format(i_k[0], i_k[1])
                               ][a] * noise_dict[
                               '{0}x{1}'.format(j_l[0], j_l[1])
                               ][a] + noise_dict[
                               '{0}x{1}'.format(i_l[0], i_l[1])
                               ][a] * noise_dict[
                               '{0}x{1}'.format(j_k[0], j_k[1])
                               ][a]
                               )
                covmat[i, j] *= pre_fact

        covmat_list.append(covmat)

    return np.array(covmat_list)


def pre_calculation(cosmo_parameters, fg_parameters, ell_max, freq_list,
                    noise_data_path, save_path, names, n_split):

    if not os.path.isdir(save_path):
        os.mkdir(save_path)


    deriv = get_cl_derivatives(cosmo_parameters, fg_parameters,
                                   ell_max, freq_list)

    for i, name in enumerate(names):
        np.save(os.path.join(save_path, 'deriv_' + name), deriv[i])

    covariance_matrix = get_covariance_matrix(cosmo_parameters, fg_parameters,
                                              ell_max, freq_list,
                                              noise_data_path,n_split)
    np.save(os.path.join(save_path, 'covariance_matrix'), covariance_matrix)
