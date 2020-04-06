import numpy as np
import toolbox
import mflike as mfl
import os
from cobaya.model import get_model

path_to_mcmc_data = '/sps/planck/Users/tlouis/development/test_adrien/scripts/mcmc_precalc/'

data = np.load(os.path.join(path_to_mcmc_data, 'data_sim.npy'))

covariances = np.load(os.path.join(path_to_mcmc_data, 'covariance.npy'))

lmax = 4500
lmin = 2
bin_width = 20

frequency_list = [93, 145, 225]

def minus_log_like(a_tSZ, a_kSZ, a_p, beta_p, a_c, beta_c, n_CIBC, a_s, T_d,
                   _theory = {
                   "Cl": {"tt": lmax, "ee": lmax, "te": lmax}}):
    # Theory of power spectra
    ls = np.arange(lmin, lmax)
    c_tt_th = _theory.get_Cl(ell_factor = True)["tt"][lmin:lmax]
    c_ee_th = _theory.get_Cl(ell_factor = True)["ee"][lmin:lmax]
    c_te_th = _theory.get_Cl(ell_factor = True)["te"][lmin:lmax]

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

    # Compute the likelihood
    minloglike_ell = [(data[ell] - model_list[ell]).dot(
                      covariances[ell].dot(data[ell] - model_list[ell])) for ell in range(n_bin)]

    return np.sum(minloglike_ell)
