import numpy as np
import os
import toolbox
import mflike as mfl
import camb

def get_noise_dict(freq_list, noise_data_path, ell_max, bin_width):

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

def get_cl_dict(cosmo_parameters, fg_parameters, ell_max, freq_list, bin_width):

    fg_norm = {"nu_0": 150.0, "ell_0": 3000, "T_CMB": 2.725}
    components = {'tt': ["cibc", "cibp", "kSZ", "radio", "tSZ"],
                  'ee': [], 'te': []}
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

    pars = camb.CAMBparams(NonLinearModel=camb.nonlinear.Halofit(halofit_version="mead"))
    pars.set_cosmology(cosmomc_theta=cosmo_parameters[0],
                       ombh2=cosmo_parameters[1],
                       omch2=cosmo_parameters[2],
                       mnu=0.06,
                       omk=0,
                       tau=cosmo_parameters[5],
                       YHe=0.24,
                       bbn_predictor="PArthENoPE_880.2_standard.dat",
                       standard_neutrino_neff=3.046,
                       num_massive_neutrinos=1)

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

    return c_ell_dict

def get_covariance_matrix(cosmo_parameters, fg_parameters, ell_max, freq_list,
                          noise_data_path, n_split, bin_width, fsky):


    power_spectra = get_cl_dict(cosmo_parameters, fg_parameters,
                                  ell_max, freq_list, bin_width)
    cov_shape = len(power_spectra)
    noise_dict = get_noise_dict(freq_list, noise_data_path, ell_max,
                                bin_width)

    ell_list = toolbox.get_binned_list(np.arange(ell_max+1)[2:], bin_width)


    N = n_split*(n_split-1)
    covmat_dict = dict()


    for i, key1 in enumerate(power_spectra):
        for j, key2 in enumerate(power_spectra):
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


                pre_fact = 1 / (2 * ell_list + 1) / bin_width / fsky



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

    return(power_spectra, covmat_dict)


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
    inv_cov_mat_list = np.array([np.linalg.inv(
                                 element) for element in cov_mat_list])
    return(data_sims, inv_cov_mat_list)

################################################################################
################################################################################

save_mcmc_path = 'mcmc_precalc/'
if not os.path.isdir(save_mcmc_path):
    os.makedirs(save_mcmc_path)

planck_parameters = [0.0104101, 0.02242, 0.11933, 3.047, 0.9665, 0.0561]
fg_parameters = [3.3, 1.66, 6.91, 2.07, 4.88, 2.2, 3.09]
frequencies = [93, 145, 225]
fsky = 0.4
n_split = 2
ell_max = 4500
bin_width = 20
noise_data_path = '../true_fisher_forecast/sim_data/noise_tot_test/'

power_spectrum, covariance = get_covariance_matrix(planck_parameters,
                                                   fg_parameters, ell_max,
                                                   frequencies,
                                                   noise_data_path, n_split,
                                                   bin_width, fsky)


sims, inv_cov_mat_list = get_data_sims_cov(frequencies, power_spectrum, covariance)

np.save(os.path.join(save_mcmc_path, 'data_sim'), sims)
np.save(os.path.join(save_mcmc_path, 'inv_covariance'), inv_cov_mat_list)
