import os

#path = "/home/pspipe/workspace/PSpipe/project/maps2params/test/"
noise_data_path = 'sim_data/noise_tot_test/'
save_path_fig = 'saves/figures/'
save_path_dat = 'saves/datas/'
cl_path = 'pre_calc/'

if not os.path.isdir(save_path_dat):
    os.makedirs(save_path_dat)
if not os.path.isdir(save_path_fig):
    os.makedirs(save_path_fig)
if not os.path.isdir(cl_path):
    os.makedirs(cl_path)

name_param_corrmat = [
    r'$H_0$', r'$\Omega_bh^2$', r'$\Omega_ch^2$', r'$ln(10^{10}A_s)$',
    r'$n_s$', r'$\tau$', r'$a_{tSZ}$', r'$a_{kSZ}$', r'$a_p$', r'$\beta_p$',
    r'$a_c$', r'$\beta_c$', r'$a_s$'
]

names = [
    'H0', 'Ombh2', 'Omch2', 'As', 'ns', 'tau', 'atSZ', 'akSZ', 'ap', 'betap',
    'ac', 'betac', 'amp_s'
]

planck_parameters = [67.4, 0.02207, 0.1196, 3.098, 0.9616, 0.097]

fg_parameters = [3.3, 1.66, 6.91, 2.07, 4.88, 2.2, 3.09]
frequency_list = [145, 93, 225, 280, 27, 39]

fsky = 0.4
ell_max = 4500

################################################################################

calculate_data = False
plot_correlation = False
plot_fisher = False
plot_cosmo_parameters = True

if calculate_data:
    from pre_calc_corr import pre_calculation
    pre_calculation(planck_parameters, fg_parameters, ell_max, frequency_list,
                    noise_data_path, cl_path, names)

if plot_correlation:
    from corrmat_tools import corrmat_evol
    corrmat_evol(frequency_list, name_param_corrmat, save_path_fig,
                 save_path_dat, fsky, names, cl_path)

if plot_fisher:
    from corrmat_tools import fisher_norm_evol
    fisher_norm_evol(frequency_list, name_param_corrmat, save_path_fig, fsky,
                     names, cl_path)

if plot_cosmo_parameters:
    from corrmat_tools import cosmo_parameters_variation
    cosmo_parameters_variation(planck_parameters, fg_parameters,
                               frequency_list, name_param_corrmat,
                               save_path_fig, save_path_dat)
