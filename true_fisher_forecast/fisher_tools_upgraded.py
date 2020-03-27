import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle
import toolbox

def construct_covmat_data_vector(mode, names, path_pre_calc, binned):

    deriv_vect_list = []
    cov_mat_list = []

    if binned:

        covariance = pickle.load(
                     open(os.path.join(
                     path_pre_calc, 'covariance_binned.p'), 'rb'))

        deriv = pickle.load(
                open(os.path.join(
                path_pre_calc, 'deriv_binned.p'), 'rb'))

        power_spectrum = pickle.load(
                         open(os.path.join(
                         path_pre_calc, 'power_spectrums_binned.p'), 'rb'))

    else:

        covariance = pickle.load(
                     open(os.path.join(
                     path_pre_calc, 'covariance.p'), 'rb'))

        deriv = pickle.load(
                open(os.path.join(
                path_pre_calc, 'deriv.p'), 'rb'))

        power_spectrum = pickle.load(
                         open(os.path.join(
                         path_pre_calc,'power_spectrums.p'), 'rb'))


    if mode == 'tt' or mode == 'ee' or mode == 'te':
        power_spectrums = dict()
        for key in power_spectrum:
            if key[0] == mode:
                freqs = key[1].split('x')
                freq_list = [int(element) for element in freqs]
                if freq_list[0]<=freq_list[1]:
                    power_spectrums[key] = power_spectrum[key]

    if mode == 'all':
        power_spectrums = dict()
        for key in power_spectrum:
            if not key[0] == 'et':
                freqs = key[1].split('x')
                freq_list = [int(element) for element in freqs]
                if freq_list[0]<=freq_list[1]:
                    power_spectrums[key] = power_spectrum[key]


    n_ell = len(covariance['tt','tt','27x27','27x27'])
    n_freqs = len(power_spectrums)

    for ell in range(n_ell):
        cov_matrix = np.zeros((n_freqs,n_freqs))
        deriv_temp = []
        for i, key1 in enumerate(power_spectrums):
            for j, key2 in enumerate(power_spectrums):
                mode1 = key1[0]
                mode2 = key2[0]
                freq1 = key1[1]
                freq2 = key2[1]
                cov_matrix[i, j] = covariance[mode1, mode2,
                                              freq1, freq2][ell]
            deriv_temp.append([deriv[key1, element][ell] for element in names])
        cov_mat_list.append(cov_matrix)
        deriv_vect_list.append(deriv_temp)

    return(np.array(deriv_vect_list), np.array(cov_mat_list))






def compute_fisher(mode, fsky, names, path_pre_calc, binned):

    deriv, covariance_matrix = construct_covmat_data_vector(mode, names,
                                                            path_pre_calc,
                                                            binned)
    print(np.shape(covariance_matrix[0]))
    print((toolbox.cov2corr(covariance_matrix[0],remove_diag=False)))
    print(np.linalg.eigvals(covariance_matrix[0]))

    if mode == 'ee' or mode == 'te':
        n_params = len(names[:6])
    if mode == 'tt' or mode == 'all':
        n_params = len(names)

    n_ell = len(covariance_matrix)
    fisher = np.zeros((n_params, n_params))

    start_time = time.time()

    for ell in range(n_ell):
        #print(ell)
        inverse_cov_mat = np.linalg.inv(covariance_matrix[ell])
        for i in range(n_params):
            for j in range(n_params):
                first_term = deriv[ell][:, i]
                second_term = deriv[ell][:, j]

                n_cross_freq = len(first_term)
                first_term = first_term.reshape((1, n_cross_freq))
                second_term = second_term.reshape((n_cross_freq, 1))

                mat_prod = first_term.dot(inverse_cov_mat.dot(second_term))
                fisher[i, j] += fsky * mat_prod

    print('Construction de Fisher : %s secondes' % (time.time() - start_time))


    return fisher


def constraints(mode, fsky, names, path_pre_calc, save_path_dat, binned):

    for element in mode:

        fisher = compute_fisher(element, fsky, names, path_pre_calc, binned)
        print(np.linalg.eigvals(fisher))
        covar = np.linalg.inv(fisher)
        sig = np.sqrt(np.diagonal(covar))

        if binned:
            np.savetxt(os.path.join(
                       save_path_dat,'sigmas_{0}_binned.dat'.format(element)),
                       sig)
        else:
            np.savetxt(os.path.join(
                       save_path_dat,'sigmas_{0}.dat'.format(element)), sig)



def forecast_true_fisher(mode, cosmo_parameters, fg_parameters, name_param,
                         save_path_fig, save_path_dat, binned):

    fig = plt.figure(figsize=(24, 13.5))
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    colors = dict()
    colors['tt'] = 'darkred'
    colors['ee'] = 'darkblue'
    colors['te'] = 'darkgreen'
    colors['all'] = 'k'
    sigmas = dict()

    for element in mode:
        if binned:
            sigmas_temp = np.loadtxt(
                          save_path_dat + "sigmas_{0}_binned.dat".format(
                          element))
        else:
            sigmas_temp = np.loadtxt(
                          save_path_dat + "sigmas_{0}.dat".format(
                          element))
        sigmas[element] = sigmas_temp

    for i, cosmo_parameter in enumerate(cosmo_parameters):
        ax = fig.add_subplot(231 + i)
        ax.grid(True, linestyle='--')

        x = np.linspace(cosmo_parameter - 4 * sigmas[mode[0]][i],
                        cosmo_parameter + 4 * sigmas[mode[0]][i], 500)
        for element in mode:
            sigma = sigmas[element][i]
            y = toolbox.gaussian(x, cosmo_parameter, sigma)

            ax.plot(x, y / np.max(y), label = element,
                    color = colors[element])

        ax.set_title(name_param[i], fontsize=20)
        handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=18)
    fig.tight_layout()
    if binned:
        fig.savefig(os.path.join(save_path_fig, "forecast_binned.png"), dpi=300)
    else:
        fig.savefig(os.path.join(save_path_fig, "forecast.png"), dpi=300)

    fig = plt.figure(figsize=(20, 20))
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)

    for i, fg_parameter in enumerate(fg_parameters):

        ax = fig.add_subplot(331 + i)
        ax.grid(True, linestyle='--')
        x = np.linspace(fg_parameter - 4 * sigmas['tt'][i+6],
                        fg_parameter + 4 * sigmas['tt'][i+6], 500)
        for element in mode:
            if element == 'tt' or element == 'all':
                sigma = sigmas[element][i+6]
                y = toolbox.gaussian(x, fg_parameter, sigma)

                ax.plot(x, y / np.max(y), label = element,
                        color = colors[element])

        ax.set_title(name_param[i + 6], fontsize=20)
        handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.6, 0.1), fontsize=24)
    fig.tight_layout()
    if binned:
        fig.savefig(os.path.join(save_path_fig, "forecast_fg_binned.png"), dpi=300)
    else:
        fig.savefig(os.path.join(save_path_fig, "forecast_fg.png"), dpi=300)
