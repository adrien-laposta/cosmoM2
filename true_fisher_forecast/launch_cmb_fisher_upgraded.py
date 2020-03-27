import os


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

name_param_fig = [
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

frequency_list = [27, 39, 93, 145, 225, 280]
### The frequency list must be in ascending order


fsky = 0.4
ell_max = 4500
n_split = 2

mode = ['tt', 'ee', 'te', 'all']

binned = True
bin_width = 20

################################################################################

calculate_data = True
create_fisher_matrix = False
plot_forecast = False

if calculate_data:
    from pre_calc_fisher_upgraded import pre_calculation
    pre_calculation(planck_parameters, fg_parameters, ell_max, frequency_list,
                    noise_data_path, cl_path, n_split, names, binned, bin_width)

if create_fisher_matrix:
    from fisher_tools_upgraded import constraints
    constraints(mode, fsky, names, cl_path, save_path_dat, binned)

if plot_forecast:
    from fisher_tools_upgraded import forecast_true_fisher
    forecast_true_fisher(mode, planck_parameters, fg_parameters,
                         name_param_fig, save_path_fig, save_path_dat, binned)
