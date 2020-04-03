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

































fiducial_params = {
    'ombh2': 0.022, 'omch2': 0.12, 'H0': 68, 'tau': 0.07,
    'As': 2.2e-9, 'ns': 0.96,
    'mnu': 0.06, 'nnu': 3.046, 'num_massive_neutrinos': 1}


l_max = 4000

modules_path = ''

info_fiducial = {
    'params': fiducial_params,
    'likelihood': {'one': None},
    'theory': {'classy': None},
    'modules': modules_path}

model_fiducial = get_model(info_fiducial)


# Declare our desired theory product
# (there is no cosmological likelihood doing it for us)
model_fiducial.likelihood.theory.needs(Cl={'tt': l_max})

# Compute and extract the CMB power spectrum
# (In muK^-2, without l(l+1)/(2pi) factor)
# notice the empty dictionary below: all parameters are fixed
model_fiducial.logposterior({})
Cls = model_fiducial.likelihood.theory.get_Cl(ell_factor=False)


# Our fiducial power spectrum
Cl_est = Cls['tt'][:l_max+1]

def my_like(
        # Parameters that we may sample over (or not)
        noise_std_pixel=20,  # muK
        beam_FWHM=0.25,  # deg
        # Declaration of our theory requirements
        _theory={'Cl': {'tt': l_max}},
        # Declaration of available derived parameters
        _derived={'Map_Cl_at_500': None}):
    # Noise spectrum, beam-corrected
    healpix_Nside=512
    pixel_area_rad = np.pi/(3*healpix_Nside**2)
    weight_per_solid_angle = (noise_std_pixel**2 * pixel_area_rad)**-1
    beam_sigma_rad = beam_FWHM / np.sqrt(8*np.log(2)) * np.pi/180.
    ells = np.arange(l_max+1)
    Nl = np.exp((ells*beam_sigma_rad)**2)/weight_per_solid_angle
    # Cl of the map: data + noise
    Cl_map = Cl_est + Nl
    # Cl from theory: treat '_theory' as a 'theory code instance'
    Cl_theo = _theory.get_Cl(ell_factor=False)['tt'][:l_max+1]  # muK-2
    Cl_map_theo = Cl_theo + Nl
    # Set our derived parameter, assuming '_derived' is a dictionary
    _derived['Map_Cl_at_500'] = Cl_map[500]
    # Auxiliary plot
    ell_factor = ells*(ells+1)/(2*np.pi)
    plt.figure()
    plt.plot(ells[2:], (Cl_theo*ell_factor)[2:], label=r'Theory $C_\ell$')
    plt.plot(ells[2:], (Cl_est*ell_factor)[2:], label=r'Estimated $C_\ell$')
    plt.plot(ells[2:], (Cl_map*ell_factor)[2:], label=r'Map $C_\ell$')
    plt.plot(ells[2:], (Nl*ell_factor)[2:], label='Noise')
    plt.legend()
    plt.ylim([0, 6000])
    plt.show()
    plt.close()
    # ----------------
    # Compute the log-likelihood
    V = Cl_map[2:]/Cl_map_theo[2:]
    return np.sum((2*ells[2:]+1)*(-V/2 +1/2.*np.log(V)))

info = {
    'params': {
        # Fixed
        'ombh2': 0.022, 'omch2': 0.12, 'H0': 68, 'tau': 0.07,
        'mnu': 0.06, 'nnu': 3.046, 'num_massive_neutrinos': 1,
        # Sampled
        'As': {'prior': {'min': 1e-9, 'max': 4e-9}, 'latex': 'A_s'},
        'ns': {'prior': {'min': 0.9, 'max': 1.1}, 'latex': 'n_s'},
        'noise_std_pixel': {
            'prior': {'dist': 'norm', 'loc': 20, 'scale': 5},
            'latex': r'\sigma_\mathrm{pix}'},
        # Derived
        'Map_Cl_at_500': {'latex': r'C_{500,\,\mathrm{map}}'}},
    'likelihood': {'my_cl_like': my_like},
    'theory': {'camb': {'stop_at_error': True}},
    'sampler': {'mcmc': None},  # or polychord...
    'modules': modules_path,
    'output': 'chains/my_imaginary_cmb'}



# Activate timing (we will use it later)
info['timing'] = True

from cobaya.model import get_model
model = get_model(info)

As = np.linspace(1e-9, 4e-9, 10)
likes = [model.loglike({'As': A, 'ns': 0.96, 'noise_std_pixel': 20})[0] for A in As]

plt.figure()
plt.plot(As, likes)
plt.show()
info['likelihood']['my_cl_like'] = {
    'external': my_like,
    'speed': 500}

#run(info)
