import numpy as np
import toolbox
import mflike as mfl
import os
from cobaya.model import get_model

path_to_mcmc_data = 'mcmc_precalc/'

data = np.load(os.path.join(path_to_mcmc_data, 'data_sim.npy'))

covariances = np.load(os.path.join(path_to_mcmc_data, 'covariance.npy'))

lmax = 4500
lmin = 2
binwidth = 20

fg_parameters = [3.3, 1.66, 6.91, 2.07, 4.88, 2.2, 3.09]
frequency_list = [93, 145, 225]

def minus_log_like(_theory = {
                   "Cl": {"tt": lmax, "ee": lmax, "te": lmax}
                   }):
    # Theory of power spectra
    ls = np.arange(lmin, lmax)
    c_tt_th = _theory.get_cl(ell_factor = True)["tt"][lmin:lmax]
    c_ee_th = _theory.get_cl(ell_factor = True)["ee"][lmin:lmax]
    c_te_th = _theory.get_cl(ell_factor = True)["te"][lmin:lmax]

    # Import the foregrounds

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

    fg_dict = mfl.get_foreground_model(fg_params, fg_model, frequency_list,
                                           ls)

    # Create the power spectrum binned dict
    power_spectrum = {}
    for f1 in freq_list:
        for f2 in freq_list:

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

                power_spectra[key] = ps_dict[key]

    # Create the arrays of data in order to compute loglike
    n_ell = len(ls)
    n_freqs = len(power_spectra)
    model_list = []
    for ell in range(n_ell):
        model = np.zeros(n_freqs)
        for i, key in enumerate(power_spectra):
            model[i] = power_spectra[key][ell]
        model_list.append(model)
    model_list = np.array(model_list)

    # Compute the likelihood
    minloglike_ell = [(data[ell] - model[ell]).dot(
                      covariances[ell].dot(data[ell] - model[ell]))]

    return np.sum(minloglike_ell)

info = {
    'params': {
    'logA': {'prior': {'min': 1.61, 'max': 3.91}, 'latex': '\log(10^{10} A_\mathrm{s})', 'drop': True},
    'ns': {'prior': {'min': 0.8, 'max': 1.2}, 'latex': 'n_\mathrm{s})'},
    'H0': {'prior': {'min': 20, 'max': 100}, 'latex': 'H_0'},
    'tau': {'prior': {'min': 0.01, 'max': 0.8}, 'latex': '\tau_\mathrm{reio}'},
    'ombh2': {'prior': {'min': 0.005, 'max': 0.1}, 'latex': '\Omega_\mathrm{b} h^2'},
    'omch2': {'prior': {'min': 0.001, 'max': 0.99}, 'latex': '\Omega_\mathrm{c} h^2'}},



}
