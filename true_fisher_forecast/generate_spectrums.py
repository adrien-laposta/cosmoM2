import numpy as np
import sys
import os
import pickle
import toolbox
import matplotlib.pyplot as plt

def construct_covmat_data_vector(path_pre_calc, binned):

    cov_mat_list = []
    ps_vect_list = []
    if binned:

        covariance = pickle.load(
                     open(os.path.join(
                     path_pre_calc, 'covariance_binned.p'), 'rb'))

        power_spectrum = pickle.load(
                         open(os.path.join(
                         path_pre_calc, 'power_spectrums_binned.p'), 'rb'))

    else:

        covariance = pickle.load(
                     open(os.path.join(
                     path_pre_calc, 'covariance.p'), 'rb'))


        power_spectrum = pickle.load(
                         open(os.path.join(
                         path_pre_calc,'power_spectrums.p'), 'rb'))


    power_spectra = dict()
    for key in power_spectrum:
        if not key[0] == 'et':
            freqs = key[1].split('x')
            freq_list = [int(element) for element in freqs]
            if freq_list[0]<=freq_list[1]:
                power_spectra[key] = power_spectrum[key]


    n_ell = len(covariance['tt','tt','27x27','27x27'])
    n_freqs = len(power_spectra)

    for ell in range(n_ell):
        cov_matrix = np.zeros((n_freqs,n_freqs))
        power_spectr_temp = []
        for i, key1 in enumerate(power_spectra):
            for j, key2 in enumerate(power_spectra):
                mode1 = key1[0]
                mode2 = key2[0]
                freq1 = key1[1]
                freq2 = key2[1]
                cov_matrix[i, j] = covariance[mode1, mode2,
                                              freq1, freq2][ell]
            power_spectr_temp.append(power_spectra[key1][ell] )
        cov_mat_list.append(cov_matrix)
        ps_vect_list.append(power_spectr_temp)

    keys = [key for key in power_spectra]
    for i, element in enumerate(cov_mat_list):
        a = np.linalg.eigh(element)[0]
        somme = np.sum([1 for number in a if float(number)<0])
        print(somme, "multipole", i)
        if somme > 0:
            print(a)

    ###binn : ell 213-->218 +223 avec une vp négative
    ###unbinn : ell 4267 --> 4377   4461--> 4498
    np.savetxt("covar_4267.dat", cov_mat_list[4267])
    return(np.array(ps_vect_list), np.array(cov_mat_list), keys)
### ps_vect_list[ell] est le vecteur de données à ell fixé
### ps_vect_list[:][i] correspond au spectre pour la clé keys[i]

### Simulation des vecteurs de données à ell fixé :

def simu_data_vector(data_vector_mean, covariance_matrix):

    n_variables = len(data_vector_mean)

    #add =  np.linalg.cholesky(covariance_matrix).dot(
               #np.random.normal(size = n_variables))
    sqrt_mat = toolbox.svd_pow(covariance_matrix, 0.5)
    add = sqrt_mat.dot(np.random.normal(size = n_variables))
    data_vector_exp = data_vector_mean + add
    return data_vector_exp

### Itération pour simuler pour chaque valeur de ell :

def simu_data_vector_list(data_vector_mean_list, covariance_matrix_list):

    data_vector_exp_list = []
    n_ell = len(data_vector_mean_list)

    data_vector_exp_list = [
        simu_data_vector(data_vector_mean_list[ell],
                         covariance_matrix_list[ell]) for ell in range(n_ell)
                        ]
    return np.array(data_vector_exp_list)

def iterate_simulation_data_vector(data_vector_mean_list,
                                   covariance_matrix_list, N_simus):

    simulation = [simu_data_vector_list(data_vector_mean_list,
                                        covariance_matrix_list) for i in range(
                                        N_simus)]

    return np.array(simulation)

def compute_covariance_matrices(simulation_list):

    N_sims = len(simulation_list)
    n_ell = len(simulation_list[0])
    cov_list_exp = []
    for ell in range(n_ell):
        covar = np.cov(simulation_list[:, ell].T)
        cov_list_exp.append(covar)
    return np.array(cov_list_exp)

################################################################################
################################################################################

cl_path = 'pre_calc/'
binned = False
N_simus = 100


ps, covar, keys = construct_covmat_data_vector(cl_path, binned)

simulation = iterate_simulation_data_vector(ps, covar, N_simus)

covar_exp = compute_covariance_matrices(simulation)
print(covar_exp[122])
print(covar[122])
