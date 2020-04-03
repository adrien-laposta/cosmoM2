import pickle
import numpy as np
import os
import toolbox

path_to_pre_calc = 'pre_calc/'

frequencies = [93, 145, 225]

covariance = pickle.load(
             open(os.path.join(
             path_to_pre_calc, 'covariance_binned.p'), 'rb'))

power_spectrum = pickle.load(
                 open(os.path.join(
                 path_to_pre_calc, 'power_spectrums_binned.p'), 'rb'))

def get_data_sims_cov(frequency_list, ps_dict, cov_dict):
    power_spectra = {}
    for key in ps_dict:
        if not key[0] == 'et':
            freqs = key[1].split('x')
            freq_list = [int(element) for element in freqs]
            if freq_list[0] <= freq_list[1]:
                if (freq_list[0] in frequency_list) and (
                    freq_list[1] in frequency_list):

                    power_spectra[key] = ps_dict[key]
    n_ell = len(power_spectra['tt', '{0}x{1}'.format(frequency_list[0],
                                                     frequency_list[0])])
    n_freqs = len(power_spectra)
    cov_mat_list = []
    data_sims_mean = []
    for ell in range(n_ell):
        data_sim = np.zeros(n_freqs)
        cov_mat = np.zeros((n_freqs,n_freqs))
        for i, key1 in enumerate(power_spectra):
            for j, key2 in enumerate(power_spectra):
                mode1 = key1[0]
                mode2 = key2[0]
                freq1 = key1[1]
                freq2 = key2[1]
                cov_mat[i,j] = cov_dict[mode1,mode2,freq1,freq2][ell]
            data_sim[i] = ps_dict[key1][ell]
        cov_mat_list.append(cov_mat)
        data_sims_mean.append(data_sim)
    cov_mat_list = np.array(cov_mat_list)
    data_sims_mean = np.array(data_sims_mean)
    data_sims = []
    for ell in range(n_ell):
        X = np.random.normal(size = n_freqs)
        X = toolbox.svd_pow(cov_mat_list[ell], 0.5).dot(X)
        data_sims.append(data_sims_mean[ell] + X)
    data_sims = np.array(data_sims)
    return(data_sims, cov_mat_list)

save_mcmc_path = 'mcmc_precalc/'
if not os.path.isdir(save_mcmc_path):
    os.makedirs(save_mcmc_path)

sims, cov_mat_list = get_data_sims_cov(frequencies, power_spectrum, covariance)

np.save(os.path.join(save_mcmc_path, 'data_sim'), sims)
np.save(os.path.join(save_mcmc_path, 'covariance'), cov_mat_list)
