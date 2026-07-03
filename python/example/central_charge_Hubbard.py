"""Example to extract the central charge from the entanglement scaling.

This example code evaluate the central charge of the transverse field Ising model using iDMRG.
The expected value for the central charge c = 1/2. The code always recycle the environment from
the previous simulation, which can be seen at the "age".

For the theoretical background why :math:`S = c/6 log(xi)`, see :cite:`pollmann2009`.
"""
# Copyright (C) TeNPy Developers, Apache license

import time

import numpy as np

import tenpy
from tenpy.algorithms import dmrg
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinHalfFermionSite
import sys
sys.path.append('..')
import utils.io as io


class HubbardSpinDepV(CouplingMPOModel):
    """
    1D Hubbard model with spin-dependent nearest-neighbor repulsion.

    Model parameters:
        t     : float — NN hopping (default 1.0)
        U     : float — on-site Hubbard repulsion 
        Vpp   : float — V^{++} = V^{--}, same-spin NN repulsion 
        Vpm   : float — V^{+-} = V^{-+}, opposite-spin NN repulsion
                        FM condition: Vpm > Vpp
        mu    : float — chemical potential (default 0.0, half filling)
        bc_MPS    : 'infinite' or 'finite'
        L         : unit cell / chain length
    """

    def init_sites(self, model_params):
        cons_N = model_params.get("cons_N", "N")
        cons_Sz = model_params.get("cons_Sz", "None")
        return SpinHalfFermionSite(cons_N = cons_N, cons_Sz = cons_Sz)

    def init_terms(self, model_params):
        t   = model_params.get("t",   1.0)
        U   = model_params.get("U", None)
        Vpp = model_params.get("Vpp", None)   # V^{++}: same spin
        Vpm = model_params.get("Vpm", None)   # V^{+-}: opposite spin
        mu  = model_params.get("mu",  0.0)

        # NN hopping (plus_hc=True adds the Hermitian conjugate automatically)
        self.add_coupling(-t, 0, "Cdu", 0, "Cu", 1, plus_hc=True)
        self.add_coupling(-t, 0, "Cdd", 0, "Cd", 1, plus_hc=True)

        # On-site Hubbard U
        self.add_onsite(U, 0, "NuNd")

        # V^{++}: same-spin NN repulsion  (↑↑ and ↓↓)
        self.add_coupling(Vpp, 0, "Nu", 0, "Nu", 1)
        self.add_coupling(Vpp, 0, "Nd", 0, "Nd", 1)

        # V^{+-}: opposite-spin NN repulsion  (↑↓ and ↓↑)
        self.add_coupling(Vpm, 0, "Nu", 0, "Nd", 1)
        self.add_coupling(Vpm, 0, "Nd", 0, "Nu",   1)

        # Chemical potential
        if abs(mu) > 1e-12:
            self.add_onsite(-mu, 0, "Ntot")

def example_DMRG_tf_ising_infinite_S_xi_scaling(Vpm):
    model_params = dict(
        t=1.0, U=0.1, Vpp=0.8, Vpm=Vpm, mu=0.0,
        bc_MPS="infinite",
        L=2, cons_N="N", cons_Sz="None",
    )
    M = HubbardSpinDepV(model_params)
    psi = MPS.from_product_state(M.lat.mps_sites(), ["up", "empty"] , bc=M.lat.bc_MPS)
    dmrg_params = {
        'start_env': 10,
        'mixer': False,
        #  'mixer_params': {'amplitude': 1.e-3, 'decay': 5., 'disable_after': 50},
        'trunc_params': {'chi_max': 5, 'svd_min': 1.0e-10},
        'max_E_err': 1.0e-9,
        'max_S_err': 1.0e-6,
        'update_env': 0,
    }

    chi_list = np.arange(7, 31, 2)
    s_list = []
    xi_list = []
    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)

    for chi in chi_list:
        t0 = time.time()

        eng.reset_stats()
        # necessary if you for example have a fixed number of sweeps, if you don't set this you
        # option your simulation stops after initial number of sweeps!

        eng.trunc_params['chi_max'] = chi
        ##   DMRG Calculation    ##
        print('Start iDMRG CALCULATION')
        eng.run()
        eng.options['mixer'] = None
        psi.canonical_form()

        ##   Calculating bond entropy and correlation length  ##
        s_list.append(psi.entanglement_entropy()[0])
        # the bond 0 is between MPS unit cells and hence sensible even for 2D lattices.
        xi_list.append(psi.correlation_length())

        print(chi, time.time() - t0, np.mean(psi.expectation_value(M.H_bond)), s_list[-1], xi_list[-1], flush=True)
        tenpy.tools.optimization.optimize(3)  # quite some speedup for small chi

        print('SETTING NEW BOND DIMENSION')

    results = {
        "diagnostics" : eng.sweep_stats
    }
    io.save_mps_with_metadata("lastVpm_{Vpm:.3f}.h5", psi, results)
    return s_list, xi_list


def fit_plot_central_charge(s_list, xi_list, filename):
    """Plot routine in order to determine the cental charge."""
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    def fitFunc(Xi, c, a):
        return (c / 6) * np.log(Xi) + a

    Xi = np.array(xi_list)
    S = np.array(s_list)
    LXi = np.log(Xi)  # Logarithm of the correlation length xi

    fitParams, fitCovariances = curve_fit(fitFunc, Xi, S)

    # Plot fitting parameter and covariances
    print('c =', fitParams[0], 'a =', fitParams[1])
    print('Covariance Matrix', fitCovariances)

    # plot the data as blue circles
    plt.errorbar(
        LXi, S, fmt='o', c='blue', ms=5.5, markerfacecolor='white', markeredgecolor='blue', markeredgewidth=1.4
    )
    # plot the fitted line
    plt.plot(LXi, fitFunc(Xi, fitParams[0], fitParams[1]), linewidth=1.5, c='black', label=f'fit c={fitParams[0]:.2f}')

    plt.xlabel(r'$\log{\,}\xi_{\chi}$', fontsize=16)
    plt.ylabel(r'$S$', fontsize=16)
    plt.legend(loc='lower right', borderaxespad=0.0, fancybox=True, shadow=True, fontsize=16)
    plt.savefig(filename)


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)
    s_list, xi_list = example_DMRG_tf_ising_infinite_S_xi_scaling(Vpm=2.0) #Deep in para phase 
    fit_plot_central_charge(s_list, xi_list, 'central_charge_para.pdf')