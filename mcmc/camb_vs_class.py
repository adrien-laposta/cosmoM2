import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cobaya
import camb
import classy
import toolbox
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
    "nnu": 3.046,
    "num_massive_neutrinos": 1}

info = {
    "params": cosmo_params,
    "likelihood": {"one": None},
    "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}}
}


from cobaya.model import get_model
camb_model = get_model(info)

lmin, lmax = 2, 90


camb_model.likelihood.theory.needs(Cl={"tt": lmax, "ee": lmax, "te": lmax})
camb_model.logposterior({})

from copy import deepcopy
class_info = deepcopy(info)
class_info["theory"] = {"classy": None}

#class_params = {"H0": 67.243,
    #"logA": {"value": 3.06304, "drop": True},
    #"A_s": {"value": "lambda logA: 1e-10*np.exp(logA)"},
    #"omega_b": 0.022681,
    #"omega_cdm": 0.12106,
    #"n_s": 0.964197,
    #"tau_reio": 0.062026,
    #"m_ncdm": 0.06,
    #"N_ur": 2.0328,
    #"N_ncdm": 1}

class_params = {
    "H0": 67.7554,
    "logA": {"value": 3.035, "drop": True},
    "A_s": {"value": "lambda logA: 1e-10*np.exp(logA)"},
    "omega_b": 0.02256,
    "omega_cdm": 0.11945,
    "n_s": 0.9633,
    "tau_reio": 0.0492,
    "m_ncdm": 0.06,
    "N_ur": 2.046,
    "N_ncdm": 1}

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

    plt.suptitle(mode.upper())
    plt.legend()
    plt.savefig("{0}.png".format(mode), dpi=300)
