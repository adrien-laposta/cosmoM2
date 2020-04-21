import argparse
import sys
from classy import Class
import pickle
import numpy as np
import toolbox
import mflike as mfl
import os

#==============================================================================#
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help = 'choose the model with which simulate data')
parser.add_argument('-l', '--lmax', help = 'choose the max ell to compute power spectra', type = int)
args = parser.parse_args()
if args.model != "LCDM" and args.model != "EDE":
    sys.exit("ERROR : Please choose LCDM or EDE with the -m argument")
if not(args.lmax):
    sys.exit("ERROR : You have to choose a max ell with the -l argument")
print('--'*15)
print('Running ...')
print('model --- {0}'.format(args.model))
print('l_max --- {0}'.format(args.lmax))
print('--'*15)

#==================================================================#
#========================= initialization =========================#
#==================================================================#

model = args.model
ell_max = args.lmax
bin_width = 30

noise_path = '../../true_fisher_forecast/sim_data/noise_tot_test/'
save_pickle = '../pre_calc/'
save_mc = '../pre_calc/mc_input/'
if not os.path.isdir(save_mc):
    os.makedirs(save_mc)

frequency_list_pickle = [27, 39, 93, 145, 225, 280]
frequency_list_mc = [93, 145, 225]
fsky = 0.4
n_split = 2

if model == 'LCDM':
    parameters = {'output': 'tCl,pCl,lCl',
                  'lensing': 'yes',
                  '100*theta_s': 1.04200,
                  'omega_b': 0.02244,
                  'omega_cdm': 0.1201,
                  'A_s': (1e-10)*np.exp(3.055),
                  'n_s': 0.9659,
                  'tau_reio': 0.0587,
                  'm_ncdm': 0.06,
                  'N_ncdm': 1,
                  'N_ur': 2.0328,
                  'l_max_scalars': ell_max,
                  'delta_l_max': 1000,
                  'non linear': 'HMcode'}

elif model == 'EDE':
    parameters = {'100*theta_s': 1.04168,
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
                  'l_max_scalars': ell_max,
                  'delta_l_max': 1000}

nuisance = { 'a_tSZ': 3.3,
             'a_kSZ': 1.66,
             'a_p': 6.91,
             'beta_p': 2.07,
             'a_c': 4.88,
             'beta_c': 2.2,
             'n_CIBC': 1.2,
             'a_s': 3.09,
             'T_d': 9.6}

generate_pickles = False

#=========================================================================#
#=============================== functions ===============================#
#=========================================================================#

def get_noise_dict(freq_list, noise_path, ell_max, bin_width):

    noise_dict = {}
    for f1 in freq_list:
        for f2 in freq_list:
            if f1 == f2:
                noise_t_spectrum = np.loadtxt(
                    os.path.join(noise_path,
                    'noise_t_LAT_{0}xLAT_{0}.dat'.format(f1)))[:, 1]

                noise_pol_spectrum = np.loadtxt(
                    os.path.join(noise_path,
                    'noise_pol_LAT_{0}xLAT_{0}.dat'.format(f1)))[:, 1]

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
                '{0}x{1}'.format(f1,f2)] = toolbox.get_binned_list(
                np.zeros(ell_max-1), bin_width)

                noise_dict['ee',
                '{0}x{1}'.format(f1,f2)] = toolbox.get_binned_list(
                np.zeros(ell_max-1), bin_width)

            noise_dict['te',
            '{0}x{1}'.format(f1,f2)] = toolbox.get_binned_list(
            np.zeros(ell_max-1), bin_width)

            noise_dict['et',
            '{0}x{1}'.format(f1,f2)] = toolbox.get_binned_list(
            np.zeros(ell_max-1), bin_width)

    return(noise_dict)

def get_c_ell_dict(params, nuis_par, ell_max, freq_list, bin_width):

    fg_norm = {"nu_0": 150.0, "ell_0": 3000, "T_CMB": 2.725}
    components = {'tt': ["cibc", "cibp", "kSZ", "radio", "tSZ"],
                  'ee': [], 'te': []}
    fg_model = {'normalisation': fg_norm, 'components': components}
    fg_params = nuis_par

    c_ell = Class()
    c_ell.set(params)
    c_ell.compute()
    c_ell_lens = c_ell.lensed_cl()

    ell = c_ell_lens.get('ell')[2:]
    fg_dict = mfl.get_foreground_model(fg_params, fg_model, freq_list, ell)
    EE_l = 1e12 * pow(2.7255, 2) * ell * (ell + 1) / 2 / np.pi * c_ell_lens.get('ee')[2:]
    TT_l = 1e12 * pow(2.7255, 2) * ell * (ell + 1) / 2 / np.pi * c_ell_lens.get('tt')[2:]
    TE_l = 1e12 * pow(2.7255, 2) * ell * (ell + 1) / 2 / np.pi * c_ell_lens.get('te')[2:]

    n_freq = len(freq_list)
    d_ell_dict = {}

    for f1 in freq_list:
        for f2 in freq_list:

            d_ell_dict['tt', '{0}x{1}'.format(f1 ,f2)
            ] = toolbox.get_binned_list(TT_l + fg_dict[
            'tt', 'all', f1, f2], bin_width)

            d_ell_dict['ee', '{0}x{1}'.format(f1 ,f2)
            ] = toolbox.get_binned_list(EE_l + fg_dict[
            'ee', 'all', f1, f2], bin_width)

            d_ell_dict['te', '{0}x{1}'.format(f1 ,f2)
            ] = toolbox.get_binned_list(TE_l + fg_dict[
            'te', 'all', f1, f2], bin_width)

            d_ell_dict['et', '{0}x{1}'.format(f1 ,f2)
            ] = toolbox.get_binned_list(TE_l + fg_dict[
            'te', 'all', f1, f2], bin_width)

    return(d_ell_dict)

def get_covariances(params, nuis_par, ell_max, freq_list, noise_path,
                    n_split, bin_width, fsky):

    print('generating power spectra dict ...')
    power_spectra = get_c_ell_dict(params, nuis_par, ell_max, freq_list, bin_width)
    print('Done !')
    print('--'*15)
    cov_shape = len(power_spectra)
    print('generating noise dict ...')
    noise_dict = get_noise_dict(freq_list, noise_path, ell_max, bin_width)
    print('Done !')
    print('--'*15)
    ell = toolbox.get_binned_list(np.arange(ell_max + 1)[2:], bin_width)

    N = n_split * (n_split - 1)
    covmat_dict = {}
    print('generating covariances dict ...')
    for i, key1 in enumerate(power_spectra):
        for j, key2 in enumerate(power_spectra):
            if key1[0] != 'et' and key2[0] != 'et':

                cross_freq1 = key1[1].split('x')
                cross_freq2 = key2[1].split('x')

                nu_1 = int(cross_freq1[0])
                nu_2 = int(cross_freq1[1])
                nu_3 = int(cross_freq2[0])
                nu_4 = int(cross_freq2[1])

                R = key1[0][0]
                S = key1[0][1]
                X = key2[0][0]
                Y = key2[0][1]

                pre_fact = 1 / (2 * ell + 1) / bin_width / fsky

                temp = pre_fact * (
                    power_spectra[
                    '{0}{1}'.format(R, X), '{0}x{1}'.format(nu_1, nu_3)
                    ] * power_spectra[
                    '{0}{1}'.format(S, Y), '{0}x{1}'.format(nu_2, nu_4)
                    ] + power_spectra[
                    '{0}{1}'.format(R, Y), '{0}x{1}'.format(nu_1, nu_4)
                    ] * power_spectra[
                    '{0}{1}'.format(S, X), '{0}x{1}'.format(nu_2, nu_3)
                    ] + (power_spectra[
                    '{0}{1}'.format(R, X), '{0}x{1}'.format(nu_1, nu_3)
                    ] * noise_dict[
                    '{0}{1}'.format(S, Y), '{0}x{1}'.format(nu_2, nu_4)
                    ] + power_spectra[
                    '{0}{1}'.format(S, Y), '{0}x{1}'.format(nu_2, nu_4)
                    ] * noise_dict[
                    '{0}{1}'.format(R, X), '{0}x{1}'.format(nu_1, nu_3)
                    ]) + (power_spectra[
                    '{0}{1}'.format(R, Y), '{0}x{1}'.format(nu_1, nu_4)
                    ] * noise_dict[
                    '{0}{1}'.format(S, X), '{0}x{1}'.format(nu_2, nu_3)
                    ] + power_spectra[
                    '{0}{1}'.format(S, X), '{0}x{1}'.format(nu_2, nu_3)
                    ] * noise_dict[
                    '{0}{1}'.format(R, Y), '{0}x{1}'.format(nu_1, nu_4)
                    ]) + (1/N) * pow(n_split, 2) * (
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
    print('Done !')
    print('--'*15)
    return(power_spectra, noise_dict, covmat_dict)

def get_input_mcmc(freq_list, ps_dict, cov_dict):
    print('simulating data ...')
    power_spectra = {}
    for key in ps_dict:
        if not key[0] == 'et':
            freqs = key[1].split('x')
            cross_freq = [int(element) for element in freqs]
            if cross_freq[0] <= cross_freq[1]:
                if (cross_freq[0] in freq_list) and (
                    cross_freq[1] in freq_list):

                    power_spectra[key] = ps_dict[key]
    n_ell = len(power_spectra['tt', '{0}x{1}'.format(freq_list[0],
                                                     freq_list[0])])
    n_freqs = len(power_spectra)

    cov_mat_list = []
    data_sims_mean = []

    for ell in range(n_ell):
        data_sims = np.zeros(n_freqs)
        covmat_sims = np.zeros((n_freqs, n_freqs))
        for i, key1 in enumerate(power_spectra):
            for j, key2 in enumerate(power_spectra):
                mode1 = key1[0]
                mode2 = key2[0]
                freq1 = key1[1]
                freq2 = key2[1]
                covmat_sims[i, j] = cov_dict[mode1, mode2, freq1, freq2][ell]
            data_sims[i] = ps_dict[key1][ell]
        cov_mat_list.append(covmat_sims)
        data_sims_mean.append(data_sims)
    cov_mat_list = np.array(cov_mat_list)
    data_sims_mean = np.array(data_sims_mean)
    data_sims = []
    for ell in range(n_ell):
        X = np.random.normal(size = n_freqs)
        X = toolbox.svd_pow(cov_mat_list[ell], 0.5).dot(X)
        data_sims.append(data_sims_mean[ell] + X)
    data_sims = np.array(data_sims)
    inv_cov_mat_list = np.array([np.linalg.inv(
                                 element) for element in cov_mat_list])
    print('Done !')
    print('--'*15)
    return(data_sims, inv_cov_mat_list)

#=========================================================================#
#=========================================================================#
#=========================================================================#

if generate_pickles:
    ps, noise , covmat = get_covariances(parameters, nuisance, ell_max,
                                         frequency_list_pickle,
                                         noise_path, n_split, bin_width, fsky)
    pickle.dump(ps, open(
        os.path.join(save_pickle, 'power_spectra_binned_{0}_{1}.p'.format(bin_width, model)), 'wb'))
    pickle.dump(covmat, open(
        os.path.join(save_pickle, 'covariance_binned_{0}_{1}.p'.format(bin_width, model)),'wb'))
    pickle.dump(noise, open(
        os.path.join(save_pickle, 'noise_binned_{0}_{1}.p'.format(bin_width, model)),'wb'))

else:
    ps = pickle.load(open(
    os.path.join(save_pickle, 'power_spectra_binned_{0}_{1}.p'.format(bin_width, model)), 'rb'))
    covmat = pickle.load(open(
    os.path.join(save_pickle, 'covariance_binned_{0}_{1}.p'.format(bin_width, model)), 'rb'))
    noise = pickle.load(open(
    os.path.join(save_pickle, 'noise_binned_{0}_{1}.p'.format(bin_width, model)), 'rb'))

mc_input, invcov_mc = get_input_mcmc(frequency_list_mc, ps, covmat)

np.save(os.path.join(save_mc, 'data_sim_{0}_{1}'.format(bin_width, model)), mc_input)
np.save(os.path.join(save_mc, 'inv_covariance_{0}_{1}'.format(bin_width, model)), invcov_mc)
