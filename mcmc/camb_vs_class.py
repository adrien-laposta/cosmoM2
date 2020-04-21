import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cobaya
import camb
import classy
import toolbox
from cobaya.yaml import yaml_load_file
print("     Numpy :", np.__version__)
print("Matplotlib :", mpl.__version__)
print("      CAMB :", camb.__version__)
print("    Cobaya :", cobaya.__version__)



cosmo_params = {
    "H0": 67.7554,
    "logA": {"value": 3.035, "drop": True},
    "As": {"value": "lambda logA: 1e-10*np.exp(logA)"},
    "ombh2": 0.02256,
    "omch2": 0.11945,
    "ns": 0.9633,
    "tau": 0.0492,
    "mnu": 0.06,
    "standard_neutrino_neff": 3.046,
    "num_massive_neutrinos": 1,
    "YHe": 0.24
    }

#info = {
#    "params": cosmo_params,
#    "likelihood": {"one": None},
#    "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 4}}}
#}

extra_args = {"extra_args": {
              "lens_potential_accuracy": 1,
              "bbn_predictor": 'PArthENoPE_880.2_standard.dat',
              "halofit_version": 'takahashi',
              "AccuracyBoost": 1,
              "lAccuracyBoost": 1,
              "lSampleBoost": 1,
              "DoLateRadTruncation": False,
              "max_l": 5500
              }}

info = {
    "params": cosmo_params,
    "likelihood": {"one": None},
    "theory": {"camb": extra_args}
}
from cobaya.model import get_model
camb_model = get_model(info)

lmin, lmax = 2, 4500


camb_model.likelihood.theory.needs(Cl={"tt": lmax, "ee": lmax, "te": lmax})
camb_model.logposterior({})

from copy import deepcopy
class_info = deepcopy(info)
extra_args_class = {"extra_args": {
              "non linear": 'halofit',
              "lensing": 'yes',
              "accurate_lensing": 1,
              "tol_ncdm_bg": 1.e-10,
              "recfast_Nz0": 100000,
              "tol_thermo_integration": 1.e-5,
              "recfast_x_He0_trigger_delta": 0.01,
              "recfast_x_H0_trigger_delta": 0.01,
              "evolver": 0,
              "k_min_tau0": 0.002,
              "k_max_tau0_over_l_max": 3.,
              "k_step_sub": 0.015,
              "k_step_super": 0.0001,
              "k_step_super_reduction": 0.1,
              "start_small_k_at_tau_c_over_tau_h": 0.0004,
              "start_large_k_at_tau_h_over_tau_k": 0.05,
              "tight_coupling_trigger_tau_c_over_tau_h": 0.005,
              "tight_coupling_trigger_tau_c_over_tau_k": 0.008,
              "start_sources_at_tau_c_over_tau_h": 0.006,
              "l_max_g": 50,
              "l_max_pol_g": 25,
              "l_max_ur": 50,
              "l_max_ncdm": 50,
              "tol_perturb_integration": 1.e-6,
              "perturb_sampling_stepsize": 0.01,
              "radiation_streaming_approximation": 2,
              "radiation_streaming_trigger_tau_over_tau_k": 240.,
              "radiation_streaming_trigger_tau_c_over_tau": 100.,
              "ur_fluid_approximation": 2,
              "ur_fluid_trigger_tau_over_tau_k": 50.,
              "ncdm_fluid_approximation": 3,
              "ncdm_fluid_trigger_tau_over_tau_k": 51.,
              "tol_ncdm_synchronous": 1.e-10,
              "tol_ncdm_newtonian": 1.e-10,
              "l_logstep": 1.026,
              "l_linstep": 25,
              "hyper_sampling_flat": 12.,
              "hyper_sampling_curved_low_nu": 10.,
              "hyper_sampling_curved_high_nu": 10.,
              "hyper_nu_sampling_step": 10.,
              "hyper_phi_min_abs": 1.e-10,
              "hyper_x_tol": 1.e-4,
              "hyper_flat_approximation_nu": 1.e6,
              "q_linstep": 0.20,
              "q_logstep_spline": 20.,
              "q_logstep_trapzd": 0.5,
              "q_numstep_transition": 250,
              "transfer_neglect_delta_k_S_t0": 100.,
              "transfer_neglect_delta_k_S_t1": 100.,
              "transfer_neglect_delta_k_S_t2": 100.,
              "transfer_neglect_delta_k_S_e": 100.,
              "transfer_neglect_delta_k_V_t1": 100.,
              "transfer_neglect_delta_k_V_t2": 100.,
              "transfer_neglect_delta_k_V_e": 100.,
              "transfer_neglect_delta_k_V_b": 100.,
              "transfer_neglect_delta_k_T_t2": 100.,
              "transfer_neglect_delta_k_T_e": 100.,
              "transfer_neglect_delta_k_T_b": 100.,
              "neglect_CMB_sources_below_visibility": 1.e-30,
              "transfer_neglect_late_source": 3000.,
              "halofit_k_per_decade": 3000.,
              "l_switch_limber": 40.,
              "num_mu_minus_lmax": 1000.,
              "delta_l_max": 1000.
              }}
class_info["theory"] = {"classy": extra_args_class}

class_params = {
    "H0": 67.7554,
    "logA": {"value": 3.035, "drop": True},
    "A_s": {"value": "lambda logA: 1e-10*np.exp(logA)"},
    "omega_b": 0.02256,
    "omega_cdm": 0.11945,
    "n_s": 0.9633,
    "tau_reio": 0.0492,
    "m_ncdm": 0.06,
    "N_eff": 3.046,
    "N_ncdm": 1,
    "YHe": 0.24
    }

class_info["params"] = class_params
class_model = get_model(class_info)
class_model.likelihood.theory.needs(Cl={"tt": lmax, "ee": lmax, "te": lmax})
class_model.logposterior({})
Dls_class = class_model.likelihood.theory.get_Cl(ell_factor=True)
Dls_camb = camb_model.likelihood.theory.get_Cl(ell_factor=True)
Dl_tt_class, Dl_tt_camb = Dls_class["tt"][lmin:lmax], Dls_camb["tt"][lmin:lmax]
Dl_ee_class, Dl_ee_camb = Dls_class["ee"][lmin:lmax], Dls_camb["ee"][lmin:lmax]
Dl_te_class, Dl_te_camb = Dls_class["te"][lmin:lmax], Dls_camb["te"][lmin:lmax]
ls = np.arange(lmin,lmax)


Dl_class = [Dl_tt_class, Dl_ee_class, Dl_te_class]
Dl_camb = [Dl_tt_camb, Dl_ee_camb, Dl_te_camb]
modes = ["tt", "ee", "te"]


#data = np.load("mcmc_precalc/data_sim.npy")[:, 1]  #ee 93x93
#invcov = np.load("mcmc_precalc/inv_covariance.npy")
#cov = np.array([np.linalg.inv(element) for element in invcov])
#err = np.sqrt(cov[:,1,1])


#ls = toolbox.get_binned_list(ls, 20)
#Dl_class = [toolbox.get_binned_list(element, 20) for element in Dl_class]
#Dl_camb = [toolbox.get_binned_list(element, 20) for element in Dl_camb]

for i, mode in enumerate(modes):
    fig = plt.figure(figsize = (18,10))
    ax = fig.add_subplot(121)
    ax.grid(True,linestyle='dotted')
    ax.plot(ls, Dl_class[i], label="Class", color = 'darkred', lw=3)
    ax.plot(ls, Dl_camb[i], label="Camb", color = 'k', linestyle = 'dotted', lw=3)
    #if mode == "ee":
    #    ax.plot(ls, data[:len(ls)], color = "darkorange", label = "Model", linestyle = 'None', marker='o')
    #    ax.errorbar(ls, data[:len(ls)],err[:len(ls)], color="darkorange", linestyle = 'None')

    plt.legend()
    ax = fig.add_subplot(122)
    ax.grid(True, linestyle='dotted')
    ax.plot(ls, (Dl_camb[i]-Dl_class[i])/Dl_camb[i], color = 'k', lw = 3,
            label = r'$\frac{\Delta C_{\ell}}{C_{\ell}^{CAMB}}$')
    ax.set_xscale('log')
    plt.suptitle(mode.upper())
    plt.legend()
    plt.savefig("{0}.png".format(mode), dpi=300)
