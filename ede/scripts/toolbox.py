import numpy as np


def cov2corr(cov, remove_diag=True):
    d = np.sqrt(cov.diagonal())
    corr = ((cov.T / d).T) / d
    if remove_diag:
        corr -= np.identity(cov.shape[0])
    return corr


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) /
                  (2 * np.power(sig, 2.))) / (np.sqrt(2 * np.pi * sig * sig))


def Ellipse(center, a, b, t_rot):
    u, v = center
    t = np.linspace(0, 2 * np.pi, 100)
    Ell = np.array([a * np.cos(t), b * np.sin(t)])
    #u,v removed to keep the same center location
    R_rot = np.array([[np.cos(t_rot), -np.sin(t_rot)],
                      [np.sin(t_rot), np.cos(t_rot)]])
    #2-D rotation matrix
    Ell_rot = np.zeros((2, Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:, i] = np.dot(R_rot, Ell[:, i])
    return u + Ell_rot[0, :], v + Ell_rot[1, :]

def get_binned_list(liste, length_bins):
    temp = []
    for i in range(0, len(liste), length_bins):
        if i + length_bins <= len(liste):
            temp.append(np.mean(liste[i : i+length_bins]))
    return np.array(temp)

def square_root_matrix(M):

    eigvals, eigvec = np.linalg.eig(M)
    mat_diag = np.diag(eigvals)
    inv_eigvec = np.linalg.inv(eigvec)
    #print(eigvec.dot(mat_diag.dot(inv_eigvec)))
    mat_diag = np.sqrt(mat_diag)

    square_root = eigvec.dot(mat_diag.dot(inv_eigvec))

    return(square_root)

def svd_pow(A, exponent):
    E, V = np.linalg.eigh(A)
    #print(E)
    return np.einsum("...ab,...b,...cb->...ac", V, E**exponent, V)
