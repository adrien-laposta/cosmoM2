import numpy as np
import toolbox
import mflike as mfl
import os
from cobaya.model import get_model
import camb
import matplotlib.pyplot as plt

path_to_mcmc_data = 'mcmc_precalc/'

data = np.load(os.path.join(path_to_mcmc_data, 'data_sim.npy'))

inv_covariances = np.load(os.path.join(path_to_mcmc_data, 'inv_covariance.npy'))

lmax = 4500
lmin = 2
bin_width = 20

#planck_parameters = [67.4, 0.02207, 0.1196, 3.098, 0.9616, 0.097]
planck_parameters = [57.47255, 0.025364564, 0.14982677, 3.4991506, 1.0983872, -0.09328858]
fg_parameters = [3.3, 1.66, 6.91, 2.07, 4.88, 2.2, 3.09]

frequency_list = [93, 145, 225]

def minus_log_like(a_tSZ, a_kSZ, a_p, beta_p, a_c, beta_c, n_CIBC, a_s, T_d,
                   cosmo_parameters):
    # Theory of power spectra
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
    ell_max = 4500
    pars.set_for_lmax(ell_max - 1, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
    totCL = powers['total']

    c_ee_th = totCL[:, 1][2:]
    c_tt_th = totCL[:, 0][2:]
    c_te_th = totCL[:, 3][2:]
    ls = np.arange(len(c_tt_th)+2)[2:]
    # Import the foregrounds

    fg_norm = {"nu_0": 150.0, "ell_0": 3000, "T_CMB": 2.725}
    components = {'tt': ["cibc", "cibp", "kSZ", "radio", "tSZ"],
                  'ee': [], 'te': []}
    fg_model = {'normalisation': fg_norm, 'components': components}
    fg_params = {
        'a_tSZ': a_tSZ,
        'a_kSZ': a_kSZ,
        'a_p': a_p,
        'beta_p': beta_p,
        'a_c': a_c,
        'beta_c': beta_c,
        'n_CIBC': n_CIBC,
        'a_s': a_s,
        'T_d': T_d
    }

    fg_dict = mfl.get_foreground_model(fg_params, fg_model, frequency_list,
                                           ls)

    # Create the power spectrum binned dict
    power_spectrum = {}
    for f1 in frequency_list:
        for f2 in frequency_list:

            power_spectrum['tt', '{0}x{1}'.format(f1,f2)
            ] = toolbox.get_binned_list(c_tt_th + fg_dict[
            'tt', 'all', f1, f2], bin_width)

            power_spectrum['ee', '{0}x{1}'.format(f1,f2)
            ] = toolbox.get_binned_list(c_ee_th + fg_dict[
            'ee', 'all', f1, f2], bin_width)


            power_spectrum['te', '{0}x{1}'.format(f1,f2)
            ] = toolbox.get_binned_list(c_te_th + fg_dict[
            'te', 'all', f1, f2], bin_width)

    # Create the dict without f1xf2 with f2<f1
    power_spectra = {}
    for key in power_spectrum:
        freqs = key[1].split('x')
        freq_list = [int(element) for element in freqs]
        if freq_list[0] <= freq_list[1]:
            if (freq_list[0] in frequency_list) and (
                freq_list[1] in frequency_list):

                power_spectra[key] = power_spectrum[key]

    # Create the arrays of data in order to compute loglike
    n_ell = len(ls)
    binned_ell = toolbox.get_binned_list(ls, bin_width)
    n_bin = len(binned_ell)
    n_freqs = len(power_spectra)
    model_list = []
    for ell in range(n_bin):
        model = np.zeros(n_freqs)
        for i, key in enumerate(power_spectra):
            model[i] = power_spectra[key][ell]
        model_list.append(model)
    model_list = np.array(model_list)
    print(len(model_list[0]))
    # Compute the likelihood
    minloglike_ell = [-0.5*(data[ell] - model_list[ell]).dot(
                      inv_covariances[ell].dot(data[ell] - model_list[ell])) for ell in range(n_bin)]

    return (binned_ell, minloglike_ell, model_list, data)

#print(minus_log_like(3.3,1.66,6.91,2.07,4.88,2.2,1.2,3.09,9.6,planck_parameters))
plt.figure()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\chi^2/ndf$')
ls, minloglike_ell, model_list, data = minus_log_like(3.3,1.66,6.91,2.07,4.88,2.2,1.2,3.09,9.6,planck_parameters)
plt.plot(ls, np.array(minloglike_ell)/18)
print("chi2:", np.sum(np.array(minloglike_ell)/18))
plt.show()

plt.figure()
plt.xlabel(r'$\ell$')
L_model = []
L_data = []
ls_tot = []
for i in range(len(data[0])):
    L_model.append(model_list[:,i])
    L_data.append(data[:,i])
L_model = np.array(L_model)
L_data = np.array(L_data)
L_model = L_model.reshape(224*18)
L_data = L_data.reshape(224*18)
print(len(L_model))
print(len(L_data))
print(np.shape(L_data))
plt.plot(L_model)
plt.scatter(np.arange(len(L_data)),L_data,linewidth=0.5, color = 'k')
plt.yscale('log')
plt.show()
