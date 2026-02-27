import numpy as np
import warnings
from gpcam.gp_optimizer import GPOptimizer

import source_files.UtilityFunctions as UF

#Fixed reference A, B values
A_kernel = 0.0217 #good A from experiments
B_kernel = -8.2186 #good B from experiments

def FH_kernel(x1, x2, hps):
    """Matérn-3/2 Kernel augmented with additional Flory-Huggins interaction parameter distance term (and length scale)"""
    var_f, l_x, l_T, l_chi = hps[:4]

    x1 = np.atleast_2d(x1)  # (N1, 2)
    x2 = np.atleast_2d(x2)  # (N2, 2)

    dx = (x1[:, None, 0] - x2[None, :, 0]) / l_x  # shape (N1, N2)
    dT = (x1[:, None, 1] - x2[None, :, 1]) / l_T
    
    chi1 = UF.chi(x1[:, 1][:, None], A_kernel, B_kernel)
    chi2 = UF.chi(x2[:, 1][None, :], A_kernel, B_kernel)

    dchi = (chi1 - chi2) / l_chi
    
    r = np.sqrt(dx**2 + dT**2 + dchi**2)

    return var_f * (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
