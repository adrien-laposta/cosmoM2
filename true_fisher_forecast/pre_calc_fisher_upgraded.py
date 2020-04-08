import os

import camb
import numpy as np

import mflike as mfl
from pspy import so_dict

import pickle
import toolbox


def get_noise_dict(freq_list, noise_data_path, ell_max, binned, bin_width):

    noise_dict = dict()

    for f1 in freq_list:
        for f2 in freq_list:

            if f1 == f2:
                noise_t_spectrum = np.loadtxt(
                    os.path.join(noise_data_path,
                                'noise_t_LAT_{0}xLAT_{0}.dat'.format(f1)))[:, 1]

                noise_pol_spectrum = np.loadtxt(
                    os.path.join(noise_data_path,
                                'noise_pol_LAT_{0}xLAT_{0}.dat'.format(f1)))[:, 1]
                if binned:
                    noise_dict['tt',
                               '{0}x{1}'.format(f1, f2)
                              ] = toolbox.get_binned_list(
                              noise_t_spectrum[:ell_max - 1], bin_width)

                    noise_dict['ee',
                               '{0}x{1}'.format(f1, f2)
                              ] = toolbox.get_binned_list(
                              noise_pol_spectrum[:ell_max - 1], bin_width)
                else:
                    noise_dict['tt',
                               '{0}x{1}'.format(f1, f2)] = noise_t_spectrum[
                                                                       :ell_max - 1]

                    noise_dict['ee',
                               '{0}x{1}'.format(f1, f2)] = noise_pol_spectrum[
                                                                       :ell_max - 1]

            else:
                if binned:
                    noise_dict['tt',
                    '{0}x{1}'.format(f1,f2)] = toolbox.get_binned_list(
                    np.zeros(ell_max-1), bin_width)
                    noise_dict['ee',
                    '{0}x{1}'.format(f1,f2)] = toolbox.get_binned_list(
                    np.zeros(ell_max-1), bin_width)
                else:
                    noise_dict['tt', '{0}x{1}'.format(f1,f2)] = np.zeros(ell_max-1)
                    noise_dict['ee', '{0}x{1}'.format(f1,f2)] = np.zeros(ell_max-1)
            if binned:
                noise_dict['te',
                '{0}x{1}'.format(f1,f2)] = toolbox.get_binned_list(
                np.zeros(ell_max-1), bin_width)
                noise_dict['et',
                '{0}x{1}'.format(f1,f2)] = toolbox.get_binned_list(
                np.zeros(ell_max-1), bin_width)
            else:
                noise_dict['te', '{0}x{1}'.format(f1,f2)] = np.zeros(ell_max-1)
                noise_dict['et', '{0}x{1}'.format(f1,f2)] = np.zeros(ell_max-1)
    for key in noise_dict:
        print(np.shape(noise_dict[key]))
    return(noise_dict)



def get_cl_dict(cosmo_parameters, fg_parameters, ell_max, freq_list,
                binned, bin_width):

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

    pars.set_for_lmax(ell_max - 1, lens_potential_accuracy=1)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
    totCL = powers['total']

    EE = totCL[:, 1][2:]
    TT = totCL[:, 0][2:]
    TE = totCL[:, 3][2:]

    ell_list = np.arange(len(TT) + 2)[2:]


    fg_dict = mfl.get_foreground_model(fg_params, fg_model, freq_list,
                                       ell_list)

    n_freqs = len(freq_list)

    c_ell_dict = dict()

    for f1 in freq_list:
        for f2 in freq_list:

            if binned:

                c_ell_dict['tt', '{0}x{1}'.format(f1,f2)
                ] = toolbox.get_binned_list(TT + fg_dict[
                'tt', 'all', f1, f2], bin_width)

                c_ell_dict['ee', '{0}x{1}'.format(f1,f2)
                ] = toolbox.get_binned_list(EE + fg_dict[
                'ee', 'all', f1, f2], bin_width)


                c_ell_dict['te', '{0}x{1}'.format(f1,f2)
                ] = toolbox.get_binned_list(TE + fg_dict[
                'te', 'all', f1, f2], bin_width)


                c_ell_dict['et', '{0}x{1}'.format(f1,f2)
                ] = toolbox.get_binned_list(TE + fg_dict[
                'te', 'all', f1, f2], bin_width)

            else:

                c_ell_dict['tt', '{0}x{1}'.format(f1,f2)
                ] = TT + fg_dict['tt', 'all', f1, f2]

                c_ell_dict['ee', '{0}x{1}'.format(f1,f2)
                ] = EE + fg_dict['ee', 'all', f1, f2]

                c_ell_dict['te', '{0}x{1}'.format(f1,f2)
                ] = TE + fg_dict['te', 'all', f1, f2]

                c_ell_dict['et', '{0}x{1}'.format(f1,f2)
                ] = TE + fg_dict['te', 'all', f1, f2]
    for key in c_ell_dict:
        print(np.shape(c_ell_dict[key]))
    return c_ell_dict


### A REVOIR AVEC LES AUTRES FONCTIONS EN MODE DICT TT EE TE
def get_cl_derivatives(cosmo_parameters, fg_parameters, ell_max, freq_list,
                       names, binned, bin_width):

    n_params_cosmo = len(cosmo_parameters)
    n_params_fg = len(fg_parameters)

    epsilon_cosmo = np.array(cosmo_parameters) / 100
    epsilon_fg = np.array(fg_parameters) / 100
    var_temp = []
    deriv_dict = dict()

    for i in range(n_params_cosmo):
        eps = epsilon_cosmo[i] * np.eye(1, n_params_cosmo, i)
        eps = eps.flatten()
        CL_plus_dict = get_cl_dict(cosmo_parameters + eps, fg_parameters,
                                          ell_max, freq_list, binned, bin_width)
        CL_moins_dict = get_cl_dict(cosmo_parameters - eps, fg_parameters,
                                           ell_max, freq_list, binned, bin_width)
        for key in CL_plus_dict:
            der = (CL_plus_dict[key] - CL_moins_dict[key]) / (2 * epsilon_cosmo[i])
            deriv_dict[key,names[i]] = der

    for j in range(n_params_fg):
        eps = epsilon_fg[j] * np.eye(1, n_params_fg, j)
        eps = eps.flatten()
        CL_plus_dict = get_cl_dict(cosmo_parameters, fg_parameters + eps,
                                          ell_max, freq_list, binned, bin_width)
        CL_moins_dict = get_cl_dict(cosmo_parameters, fg_parameters - eps,
                                           ell_max, freq_list, binned, bin_width)
        for key in CL_plus_dict:
            der = (CL_plus_dict[key] - CL_moins_dict[key]) / (2 * epsilon_fg[j])
            deriv_dict[key,names[j+6]] = der

    return(deriv_dict)




def get_covariance_matrix(cosmo_parameters, fg_parameters, ell_max, freq_list,
                          noise_data_path, n_split, binned, bin_width, fsky):


    power_spectrums = get_cl_dict(cosmo_parameters, fg_parameters,
                                  ell_max, freq_list, binned, bin_width)
    cov_shape = len(power_spectrums)
    noise_dict = get_noise_dict(freq_list, noise_data_path, ell_max,
                                binned, bin_width)

    if binned:
        ell_list = toolbox.get_binned_list(np.arange(ell_max+1)[2:], bin_width)
    else:
        ell_list = np.arange(len(power_spectrums['tt', '27x27']) + 2)[2:]

    print("bla",np.shape(ell_list))
    N = n_split*(n_split-1)
    covmat_dict = dict()


    for i, key1 in enumerate(power_spectrums):
        for j, key2 in enumerate(power_spectrums):
            if key1[0] != 'et' and key2[0] != 'et':

                covmat_list = []
                cross_freqs1 = key1[1].split('x')
                cross_freqs2 = key2[1].split('x')

                nu_1 = int(cross_freqs1[0])
                nu_2 = int(cross_freqs1[1])
                nu_3 = int(cross_freqs2[0])
                nu_4 = int(cross_freqs2[1])

                R = key1[0][0]
                S = key1[0][1]
                X = key2[0][0]
                Y = key2[0][1]

                if binned:
                    pre_fact = 1 / (2 * ell_list + 1) / bin_width / fsky
                else:
                    pre_fact = 1 / (2 * ell_list + 1) / fsky



                temp = pre_fact * (
                    power_spectrums[
                    '{0}{1}'.format(R, X), '{0}x{1}'.format(nu_1, nu_3)
                    ] * power_spectrums[
                    '{0}{1}'.format(S, Y), '{0}x{1}'.format(nu_2, nu_4)
                    ] + power_spectrums[
                    '{0}{1}'.format(R, Y), '{0}x{1}'.format(nu_1, nu_4)
                    ] * power_spectrums[
                    '{0}{1}'.format(S, X), '{0}x{1}'.format(nu_2, nu_3)
                    ] + (power_spectrums[
                    '{0}{1}'.format(R, X), '{0}x{1}'.format(nu_1, nu_3)
                    ] * noise_dict[
                    '{0}{1}'.format(S, Y), '{0}x{1}'.format(nu_2, nu_4)
                    ] + power_spectrums[
                    '{0}{1}'.format(S, Y), '{0}x{1}'.format(nu_2, nu_4)
                    ] * noise_dict[
                    '{0}{1}'.format(R, X), '{0}x{1}'.format(nu_1, nu_3)
                    ]) + (power_spectrums[
                    '{0}{1}'.format(R, Y), '{0}x{1}'.format(nu_1, nu_4)
                    ] * noise_dict[
                    '{0}{1}'.format(S, X), '{0}x{1}'.format(nu_2, nu_3)
                    ] + power_spectrums[
                    '{0}{1}'.format(S, X), '{0}x{1}'.format(nu_2, nu_3)
                    ] * noise_dict[
                    '{0}{1}'.format(R, Y), '{0}x{1}'.format(nu_1, nu_4)
                    ]) + (1/N) * pow(n_split,2) * (
                    noise_dict[
                    '{0}{1}'.format(R, X), '{0}x{1}'.format(nu_1, nu_3)
                    ] * noise_dict[
                    '{0}{1}'.format(S, Y), '{0}x{1}'.format(nu_2, nu_4)
                    ] + noise_dict[
                    '{0}{1}'.format(R, Y), '{0}x{1}'.format(nu_1, nu_4)
                    ] * noise_dict[
                    '{0}{1}'.format(S, X), '{0}x{1}'.format(nu_2, nu_3)
                    ]
                    ))
                covmat_dict['{0}{1}'.format(R, S), '{0}{1}'.format(X, Y),
                            '{0}x{1}'.format(nu_1, nu_2),
                            '{0}x{1}'.format(nu_3, nu_4)] = temp

    return(covmat_dict)


def pre_calculation(cosmo_parameters, fg_parameters, ell_max, freq_list,
                    noise_data_path, save_path, n_split,
                    names, binned, bin_width, fsky):

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    print('\n')
    print('Creation du dictionnaire des cl')

    cl_dict = get_cl_dict(cosmo_parameters, fg_parameters, ell_max,
                          freq_list, binned, bin_width)

    print('Fin de creation du dictionnaire')
    print('\n')
    print('Creation du dictonnaire des covariances')

    cov_dict = get_covariance_matrix(cosmo_parameters, fg_parameters,
                                              ell_max, freq_list,
                                              noise_data_path, n_split,
                                              binned, bin_width, fsky)

    print('Fin de creation du dictionnaire')
    print('\n')
    print('Creation du dictionnaire des derivees')

    deriv_dict = get_cl_derivatives(cosmo_parameters, fg_parameters,
                                    ell_max, freq_list, names,
                                    binned, bin_width)

    print('Fin de creation du dictionnaire')
    if binned:

        pickle.dump(cl_dict, open(
            os.path.join(save_path, 'power_spectrums_binned.p'), 'wb'))
        pickle.dump(cov_dict, open(
            os.path.join(save_path, 'covariance_binned.p'),'wb'))
        pickle.dump(deriv_dict, open(
            os.path.join(save_path, 'deriv_binned.p'),'wb'))

    else:

        pickle.dump(cl_dict, open(
            os.path.join(save_path, 'power_spectrums.p'), 'wb'))
        pickle.dump(cov_dict, open(
            os.path.join(save_path, 'covariance.p'),'wb'))
        pickle.dump(deriv_dict, open(
            os.path.join(save_path, 'deriv.p'),'wb'))
