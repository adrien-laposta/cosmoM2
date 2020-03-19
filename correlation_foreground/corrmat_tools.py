import os
import time

import matplotlib.pyplot as plt
import numpy as np

import toolbox


def compute_fisher(freq_list, fsky, names, cl_path):

    start_time = time.time()

    deriv = [
        np.load(cl_path + "deriv_" + names[i] + ".npy")
        for i in range(len(names))
    ]
    C_ell = np.load(cl_path + "CL.npy")

    print("Importation : %s secondes" % (time.time() - start_time))

    n_ell = C_ell.shape[0]
    n_params = len(deriv)
    n_freqs = len(freq_list)
    fisher = np.zeros((n_params, n_params))

    start_time = time.time()

    for ell in range(n_ell):
        ls = np.arange(n_ell + 2)[2:]
        pre_fact = fsky * (2 * ls + 1) / 2
        inverse_c_ell = np.linalg.inv(C_ell[ell][:n_freqs, :n_freqs])
        for i in range(n_params):
            for j in range(n_params):
                trace_mat_prod = np.trace(
                    inverse_c_ell.dot(deriv[i][ell][:n_freqs, :n_freqs].dot(
                        inverse_c_ell.dot(deriv[j][ell][:n_freqs, :n_freqs]))))
                fisher[i, j] += pre_fact[ell] * trace_mat_prod

    print("Construction de Fisher : %s secondes" % (time.time() - start_time))

    #print(np.linalg.eigvals(F))
    return fisher


def constraints(freq_list, fsky, names, cl_path):

    fisher = compute_fisher(freq_list, fsky, names, cl_path)
    #print(fisher)
    print(np.linalg.eigvals(fisher))

    return np.linalg.inv(fisher)


def norm_fisher(freq_list, fsky, names, cl_path):

    fisher = compute_fisher(freq_list, fsky, names, cl_path)
    #eigen = np.linalg.eigvals(fisher)
    print(np.linalg.eigvals(fisher))
    #print("\n")
    #print("Normalized Fisher Matrix : ",norm_F)
    #print("\n")
    #print("Eigenvalues : ",eigen)
    return toolbox.cov2corr(fisher, remove_diag=False)


def corrmat_evol(freq_list, name_param, save_path_fig, save_path_dat, fsky,
                 names, cl_path):

    fig = plt.figure(figsize=(24, 13.5))
    sig_mat = []
    n_freqs = len(freq_list)
    for i in range(1, n_freqs):

        print("\n")
        print("-" * 15)
        print("Debut de la boucle a %s frequences" % (i + 1))

        start_time_boucle = time.time()
        covar = constraints(freq_list[:i + 1], fsky, names, cl_path)
        sig = np.sqrt(np.diagonal(covar))
        sig_mat.append(sig)
        np.savetxt(save_path_dat + str(freq_list[:i + 1]) + ".dat", covar)
        corr = toolbox.cov2corr(covar, remove_diag=False)
        ax = fig.add_subplot(231 + i)
        im = ax.imshow(corr, vmin=-1, vmax=+1, cmap='seismic')
        ax.set_xticks(np.arange(0, len(name_param), 1))
        ax.set_yticks(np.arange(0, len(name_param), 1))
        ax.set_xticklabels(name_param)
        ax.set_yticklabels(name_param)
        ax.set_title(r'f = ' + str(freq_list[:i + 1]))

        print("Fin de la boucle, temps d'execution : %s secondes" %
              (time.time() - start_time_boucle))
        print("-" * 15)

    cb_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    fig.colorbar(im, cax=cb_ax)
    sig_mat = np.array(sig_mat)
    np.savetxt(os.path.join(save_path_dat, "sigmas.dat"), sig_mat)
    fig.savefig(os.path.join(save_path_fig, "corrmat_var.png"), dpi=300)


def fisher_norm_evol(freq_list, name_param, save_path_fig, fsky, names,
                     cl_path):

    n_freqs = len(freq_list)
    for a in range(n_freqs):
        fig = plt.figure(figsize=(16, 16))
        plt.rc('xtick', labelsize=13.5)
        plt.rc('ytick', labelsize=13.5)
        print("\n")
        print("-" * 15)
        print("Debut de la boucle a %s frequences" % (a + 1))
        start_time_boucle = time.time()
        Fnorm = norm_fisher(freq_list[:a + 1], fsky, names, cl_path)
        ax = fig.add_subplot(111)
        ax.imshow(Fnorm, vmin=-1, vmax=+1, cmap='seismic')
        ax.set_xticks(np.arange(0, len(name_param), 1))
        ax.set_yticks(np.arange(0, len(name_param), 1))
        ax.set_xticklabels(name_param)
        ax.set_yticklabels(name_param)
        ax.set_title(r'f = ' + str(freq_list[:a + 1]), fontsize=13.5)
        for i in range(len(Fnorm)):
            for j in range(len(Fnorm)):
                ax.text(j,
                        i,
                        "{:0.3f}".format(Fnorm[i, j]),
                        horizontalalignment="center",
                        color="black",
                        fontsize=13.5)
        print("Fin de la boucle, temps d'execution : %s secondes" %
              (time.time() - start_time_boucle))
        print("-" * 15)
        fig.tight_layout()
        fig.savefig(os.path.join(save_path_fig,
                                 "fisher_norm_var{}.png".format(a)),
                    dpi=300)


def cosmo_parameters_variation(cosmo_parameters, fg_parameters, freq_list,
                               name_param, save_path_fig, save_path_dat):

    fig = plt.figure(figsize=(24, 13.5))
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    colors = ['darkred', 'darkgreen', 'darkblue']
    sigmas = np.loadtxt(save_path_dat + "sigmas.dat")
    for i, cosmo_parameter in enumerate(cosmo_parameters):
        sigma = sigmas[:, i]
        index_list = [1, 2, 4]
        sigma_3_plot = [sigma[index] for index in index_list]
        ax = fig.add_subplot(231 + i)
        ax.grid(True, linestyle='--')
        x = np.linspace(cosmo_parameter - 4 * np.max(sigma_3_plot),
                        cosmo_parameter + 4 * np.max(sigma_3_plot), 500)
        for index in index_list:
            y = toolbox.gaussian(x, cosmo_parameter, sigma[index])
            ax.plot(x,
                    y / np.max(y),
                    label=r'$N_{freq}$ = %s' % (index + 2),
                    color=colors[index // 2])
        ax.set_title(name_param[i], fontsize=20)
        handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=18)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path_fig, "forecast_var.png"), dpi=300)

    fig = plt.figure(figsize=(20, 20))
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    colors = ['darkred', 'darkgreen', 'darkblue']
    for i, fg_parameter in enumerate(fg_parameters):
        sigma = sigmas[:, i + 6]
        index_list = [1, 2, 4]
        sigma_3_plot = [sigma[index] for index in index_list]
        ax = fig.add_subplot(331 + i)
        ax.grid(True, linestyle='--')
        x = np.linspace(fg_parameter - 4 * np.max(sigma_3_plot),
                        fg_parameter + 4 * np.max(sigma_3_plot), 500)
        for index in index_list:
            y = toolbox.gaussian(x, fg_parameter, sigma[index])
            ax.plot(x,
                    y / np.max(y),
                    label=r'$N_{freq}$ = %s' % (index + 2),
                    color=colors[index // 2])
        ax.set_title(name_param[i + 6], fontsize=20)
        handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.6, 0.1), fontsize=24)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path_fig, "forecast_var_fg.png"), dpi=300)
