import os
import time

import matplotlib.pyplot as plt
import numpy as np

import toolbox


def compute_fisher(fsky, names, path_pre_calc):

    start_time = time.time()

    deriv = [
        np.load(path_pre_calc + 'deriv_' + names[i] + '.npy')
        for i in range(len(names))
    ]

    covariance_matrix = np.load(path_pre_calc + 'covariance_matrix.npy')
    print('Importation : %s secondes' % (time.time() - start_time))

    n_ell = len(covariance_matrix)
    n_params = len(deriv)

    fisher = np.zeros((n_params, n_params))

    start_time = time.time()
    ls = np.arange(n_ell + 2)[2:]

    for ell in range(n_ell):
        inverse_cov_mat = np.linalg.inv(covariance_matrix[ell])
        for i in range(n_params):
            for j in range(n_params):
                first_term = deriv[i][:, ell]
                second_term = deriv[j][:, ell]

                n_cross_freq = len(first_term)
                first_term = first_term.reshape((1, n_cross_freq))
                second_term = second_term.reshape((n_cross_freq, 1))

                mat_prod = first_term.dot(inverse_cov_mat.dot(second_term))
                fisher[i, j] += fsky * mat_prod

    print('Construction de Fisher : %s secondes' % (time.time() - start_time))

    #print(np.linalg.eigvals(F))
    return fisher


def constraints(fsky, names, path_pre_calc, save_path_dat):

    fisher = compute_fisher(fsky, names, path_pre_calc)
    #print(fisher)
    print(np.linalg.eigvals(fisher))
    covar = np.linalg.inv(fisher)
    sig = np.sqrt(np.diagonal(covar))
    np.savetxt(os.path.join(save_path_dat,'sigmas.dat'),sig)
    return covar


def forecast_true_fisher(cosmo_parameters, fg_parameters, name_param,
                         save_path_fig, save_path_dat):

    fig = plt.figure(figsize=(24, 13.5))
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    sigmas = np.loadtxt(save_path_dat + "sigmas.dat")
    sigma_fisher_old = [9.26778e-01, 1.21609e-04, 2.33866e-03, 3.55678e-02,
                        5.41443e-03, 1.98027e-02, 2.70764e-02, 8.18402e-02,
                        9.08642e-02, 1.09348e-02, 9.58081e-02, 1.60091e-02,
                        3.76902e-03]

    for i, cosmo_parameter in enumerate(cosmo_parameters):
        sigma_old = sigma_fisher_old[i]
        sigma = sigmas[i]

        ax = fig.add_subplot(231 + i)
        ax.grid(True, linestyle='--')

        x = np.linspace(cosmo_parameter - 4 * sigma,
                        cosmo_parameter + 4 * sigma, 500)

        y = toolbox.gaussian(x, cosmo_parameter, sigma)
        y_old = toolbox.gaussian(x,cosmo_parameter, sigma_old)
        ax.plot(x, y / np.max(y), label="CMB Likelihood Fisher",
                color='darkblue')
        ax.plot(x, y_old / np.max(y_old), label="Old Fisher",
                color='darkred')

        ax.set_title(name_param[i], fontsize=20)
        handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=18)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path_fig, "forecast.png"), dpi=300)

    fig = plt.figure(figsize=(20, 20))
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)

    for i, fg_parameter in enumerate(fg_parameters):
        sigma_old = sigma_fisher_old[i+6]
        sigma = sigmas[i+6]
        ax = fig.add_subplot(331 + i)
        ax.grid(True, linestyle='--')
        x = np.linspace(fg_parameter - 4 * sigma,
                        fg_parameter + 4 * sigma, 500)
        y = toolbox.gaussian(x, fg_parameter, sigma)
        y_old = toolbox.gaussian(x,fg_parameter,sigma_old)
        ax.plot(x, y / np.max(y), label="CMB Likelihood Fisher",
                color='darkblue')
        ax.plot(x, y_old / np.max(y_old), label="Old Fisher",
                color='darkred')
        ax.set_title(name_param[i + 6], fontsize=20)
        handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.6, 0.1), fontsize=24)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path_fig, "forecast_fg.png"), dpi=300)
