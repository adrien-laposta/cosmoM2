import numpy as np
import toolbox
import random
import pickle
import os
import matplotlib.pyplot as plt
#ps = np.loadtxt("test_ps.dat")
#covar = np.loadtxt("test_cov.dat")
#d = len(ps)
#n = 100
#X = np.random.normal(size=(d,n))
#X -= np.mean(X)
#X = np.linalg.inv(np.linalg.cholesky(np.cov(X))).dot(X)
#X = np.linalg.cholesky(covar).dot(X)
#X = toolbox.svd_pow(covar,0.5).dot(X)
#print(covar)
#print(np.cov(X))
#covarbis = np.loadtxt("covar_4267.dat")
#print(np.diagonal(covarbis))
#print(np.sum(np.diagonal(covarbis)))
#print(np.sum(covarbis))

cl_path = 'pre_calc/'
save_path = 'recover_covar_test/'
covariance = pickle.load(
             open(os.path.join(
             cl_path, 'covariance_binned.p'), 'rb'))

power_spectrum = pickle.load(
                 open(os.path.join(
                 cl_path, 'power_spectrums_binned.p'), 'rb'))

frequencies = [93, 145, 225]


def get_covmat_list(frequency_list, ps_dict, true_cov_dict, nsims):
    power_spectra = {}
    for key in ps_dict:
        if not key[0] == 'et':
            freqs = key[1].split('x')
            freq_list = [int(element) for element in freqs]
            if freq_list[0]<=freq_list[1]:
                if freq_list[0] in frequency_list and freq_list[1] in frequency_list:
                    power_spectra[key] = ps_dict[key]

    n_ell = len(power_spectra['tt','145x145'])
    n_freqs = len(power_spectra)
    cov_mat_list = []
    for ell in range(n_ell):
        cov_mat = np.zeros((n_freqs,n_freqs))
        for i, key1 in enumerate(power_spectra):
            for j, key2 in enumerate(power_spectra):
                mode1 = key1[0]
                mode2 = key2[0]
                freq1 = key1[1]
                freq2 = key2[1]
                cov_mat[i,j] = true_cov_dict[mode1,mode2,freq1,freq2][ell]
        cov_mat_list.append(cov_mat)
    cov_mat_list = np.array(cov_mat_list)

    data_sims = []
    for ell in range(n_ell):
        X = np.random.normal(size=(n_freqs,nsims))
        X = toolbox.svd_pow(cov_mat_list[ell],0.5).dot(X)
        data_sims.append(X)
    data_sims = np.array(data_sims)
    covar_exp = np.array([np.cov(data_sims[ell]) for ell in range(n_ell)])

    return(covar_exp, cov_mat_list, power_spectra)


covar_exp_1000, covar_th, power_spectra = get_covmat_list(frequencies, power_spectrum, covariance, 1000)
covar_exp_10000 = get_covmat_list(frequencies, power_spectrum, covariance, 10000)[0]
covar_exp_100000 = get_covmat_list(frequencies, power_spectrum, covariance, 100000)[0]
a = 0
for i, key1 in enumerate(power_spectra):
    for j, key2 in enumerate(power_spectra):
        if j<=i:
            plt.figure(figsize=(10,5.6))
            plt.suptitle('({0}-{1})_({2}-{3})'.format(key1[0],key1[1],key2[0],key2[1]))
            plt.plot(covar_th[:,i,j], color = 'k', label = 'Input Covariance',zorder=6, linestyle='--')
            plt.plot(covar_exp_1000[:,i,j], color= 'darkred', label = 'Recovered 1000it')
            plt.plot(covar_exp_10000[:,i,j], color= 'darkblue', label = 'Recovered 10000it')
            plt.plot(covar_exp_100000[:,i,j], color= 'darkgreen', label = 'Recovered 100000it')
            plt.grid(True,linestyle='dotted')
            plt.xlabel(r'$\ell$')
            plt.legend()
            plt.savefig(save_path + 'plot_%03d.png'%a, dpi=300)
            plt.close('all')
            a+=1
