"""
Some utility functions for the map2parameters project.
"""
import numpy as np, healpy as hp
from pixell import curvedsky

def symmetrize(mat):
    """Make a upper diagonal or lower diagonal matrix symmetric
        
    Parameters
    ----------
    mat : 2d array
      the matrix we want symmetric
    """
    return mat + mat.T - np.diag(mat.diagonal())

def get_noise_matrix_spin0and2(noise_data_dir, exp, freqs, lmax, nsplits, lcut=0):
    
    """This function uses the noise power spectra computed by 'maps_to_params_prepare_sim_data'
    and generate a three dimensional array of noise power spectra [nfreqs,nfreqs,lmax] for temperature
    and polarisation.
    The different entries ([i,j,:]) of the arrays contain the noise power spectra
    for the different frequency channel pairs.
    for example nl_array_t[0,0,:] =>  nl^{TT}_{f_{0},f_{0}),  nl_array_t[0,1,:] =>  nl^{TT}_{f_{0},f_{1})
    this allows to have correlated noise between different frequency channels.
    
    Parameters
    ----------
    noise_data_dir : string
      the folder containing the noise power spectra
    exp : string
      the experiment to consider ('LAT' or 'Planck')
    freqs: 1d array of string
      the frequencies we consider
    lmax: integer
      the maximum multipole for the noise power spectra
    n_splits: integer
      the number of data splits we want to simulate
      nl_per_split= nl * n_{splits}
    lcut: integer
      the noise of SO being very red we will cut out the lowest modes to avoid
      leakage
    """

    nfreqs = len(freqs)
    nl_array_t = np.zeros((nfreqs, nfreqs, lmax))
    nl_array_pol = np.zeros((nfreqs, nfreqs, lmax))
    
    for c1, freq1 in enumerate(freqs):
        for c2, freq2 in enumerate(freqs):
            if c1>c2 : continue
            
            l, nl_t = np.loadtxt("%s/noise_t_%s_%sx%s_%s.dat"%(noise_data_dir, exp, freq1, exp, freq2), unpack=True)
            l, nl_pol = np.loadtxt("%s/noise_pol_%s_%sx%s_%s.dat"%(noise_data_dir, exp, freq1, exp, freq2), unpack=True)
            
            nl_array_t[c1, c2, lcut:lmax] = nl_t[lcut - 2:lmax - 2] * nsplits
            nl_array_pol[c1, c2, lcut:lmax] = nl_pol[lcut - 2:lmax - 2] * nsplits

    for i in range(lmax):
        nl_array_t[:,:,i] = symmetrize(nl_array_t[:,:,i])
        nl_array_pol[:,:,i] = symmetrize(nl_array_pol[:,:,i])

    return l, nl_array_t, nl_array_pol


def get_foreground_matrix(foreground_data_dir, fg_components, all_freqs, lmax):
    
    """This function uses the foreground power spectra generated by 'maps_to_params_prepare_sim_data'
    and generate a three dimensional array of foregroung power spectra [nfreqs,nfreqs,lmax].
    The different entries ([i,j,:]) of the array contains the fg power spectra for the different
    frequency channel pairs.
    for example fl_array_T[0,0,:] =>  fl_{f_{0},f_{0}),  fl_array_T[0,1,:] =>  fl_{f_{0},f_{1})
    this allows to have correlated fg between different frequency channels.
    (Not that for now, no fg are including in pol)
        
    Parameters
    ----------
    foreground_data_dir : string
      the folder containing the foreground power spectra
    fg_components: list of string
      the list of the foreground components we wish to include
    all_freqs: 1d array of string
      the frequencies we consider
    lmax: integer
      the maximum multipole for the noise power spectra
    """

    nfreqs = len(all_freqs)
    fl_array = np.zeros((nfreqs, nfreqs, lmax))
    
    for c1, freq1 in enumerate(all_freqs):
        for c2, freq2 in enumerate(all_freqs):
            if c1 > c2 : continue
            
            fl_all = 0
            for fg_comp in fg_components:
                l, fl = np.loadtxt("%s/tt_%s_%sx%s.dat"%(foreground_data_dir, fg_comp, freq1, freq2), unpack=True)
                fl_all += fl * 2 * np.pi / (l * (l + 1))
            
            fl_array[c1, c2, 2:lmax] = fl_all[:lmax-2]

    for i in range(lmax):
        fl_array[:,:,i] = symmetrize(fl_array[:,:,i])

    return l, fl_array

def multiply_alms(alms, bl, ncomp):
    
    """This routine mutliply the alms by a function bl
        
    Parameters
    ----------
    alms : 1d array
      the alms to be multiplied
    bl : 1d array
      the function to multiply the alms
    ncomp: interger
      the number of components
      ncomp = 3 if T,Q,U
      ncomp = 1 if T only
    """
    
    alms_mult = alms.copy()
    if ncomp == 1:
        alms_mult = hp.sphtfunc.almxfl(alms_mult, bl)
    else:
        for i in range(ncomp):
            alms_mult[i] = hp.sphtfunc.almxfl(alms_mult[i], bl)
    return alms_mult


def generate_noise_alms(nl_array_t, lmax, n_splits, ncomp, nl_array_pol=None):
    
    """This function generates the alms corresponding to the noise power spectra matrices
    nl_array_t, nl_array_pol. The function returns a dictionnary nlms["T", i].
    The entry of the dictionnary are for example nlms["T", i] where i is the index of the split.
    note that nlms["T", i] is a (nfreqs, size(alm)) array, it is the harmonic transform of
    the noise realisation for the different frequencies.
    
    Parameters
    ----------
    nl_array_t : 3d array [nfreq, nfreq, lmax]
      noise power spectra matrix for temperature data
    
    lmax : integer
      the maximum multipole for the noise power spectra
    n_splits: integer
      the number of data splits we want to simulate
    ncomp: interger
      the number of components
      ncomp = 3 if T,Q,U
      ncomp = 1 if T only
    nl_array_pol : 3d array [nfreq, nfreq, lmax]
      noise power spectra matrix for polarisation data
      (in use if ncomp==3)
    """
    
    nlms = {}
    if ncomp == 1:
        for k in range(n_splits):
            nlms[k] = curvedsky.rand_alm(nl_array_t,lmax=lmax)
    else:
        for k in range(n_splits):
            nlms["T", k] = curvedsky.rand_alm(nl_array_t, lmax=lmax)
            nlms["E", k] = curvedsky.rand_alm(nl_array_pol, lmax=lmax)
            nlms["B", k] = curvedsky.rand_alm(nl_array_pol, lmax=lmax)
    
    return nlms

def remove_mean(so_map, window, ncomp):
    
    """This function removes the mean value of the map after having applied the
    window function
    Parameters
    ----------
    so_map : so_map
      the map we want to subtract the mean from
    window : so_map or so_map tuple
      the window function, if ncomp=3 expect
      (win_t,win_pol)
    ncomp : integer
      the number of components
      ncomp = 3 if T,Q,U
      ncomp = 1 if T only
      
     """
    
    if ncomp == 1:
        so_map.data -= np.mean(so_map.data * window.data)
    else:
        so_map.data[0] -= np.mean(so_map.data[0] * window[0].data)
        so_map.data[1] -= np.mean(so_map.data[1] * window[1].data)
        so_map.data[2] -= np.mean(so_map.data[2] * window[1].data)

    return so_map


def get_effective_noise(nl_file_t, bl1, bl2, lmax,nl_file_pol=None, lcut=0):
    """This function returns the effective noise power spectrum
        which is defined as the ratio of the noise power spectrum and the
        beam window function.
        Parameters
        ----------
        nl_file_t: string
          name of the temperature noise power spectrum file, expect l,nl
        nl_file_p: string
          name of the polarisation noise power spectrum file, expect l,nl
        bl1: 1 d array
          beam harmonic transform
        bl2: 1 d array
          beam harmonic transform
        lmax : integer
          the maximum multipole for the noise power spectra
        lcut : integer
          the noise of SO being very red we will cut out the lowest modes to avoid
          leakage
    """

    bl1 = bl1[lcut:lmax]
    bl2 = bl2[lcut:lmax]
    if nl_file_pol == None:
        l, noise = np.loadtxt(nl_file_t, unpack=True)
        noise[lcut:lmax] /= (bl1 * bl2)
    else:
        noise = {}
        spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
        for spec in spectra:
            noise[spec] = np.zeros(lmax)
        
        l, noise_t = np.loadtxt(nl_file_t,unpack=True)
        l, noise_pol = np.loadtxt(nl_file_pol,unpack=True)

        noise["TT"][lcut:lmax] = noise_t[lcut:lmax] / (bl1 * bl2)
        noise["EE"][lcut:lmax] = noise_pol[lcut:lmax] / (bl1 * bl2)
        noise["BB"][lcut:lmax] = noise_pol[lcut:lmax] / (bl1 * bl2)
    
    return noise

def is_symmetric(mat, tol=1e-8):
    return np.all(np.abs(mat-mat.T) < tol)

def is_pos_def(mat):
    return np.all(np.linalg.eigvals(mat) > 0)

