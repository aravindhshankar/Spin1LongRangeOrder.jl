import time

import numpy as np

import tenpy
from tenpy.algorithms import dmrg
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinHalfFermionSite
import sys, os
sys.path.append('..')
import utils.io as io

# ROOTDIR = '../../data/iDMRG/Vpm3.8cc/'
# ROOTDIR = '../../data/iDMRG/Vpm6.0cc/'
# os.makedirs(ROOTDIR, exist_ok=True)

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
        hx  = model_params.get("hx", 0.0)
        hz  = model_params.get("hz", 0.0)

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

        self.add_onsite(hx, 0, "Sx")
        self.add_onsite(hz, 0, "Sz")

        # Chemical potential
        if abs(mu) > 1e-12:
            self.add_onsite(-mu, 0, "Ntot")

def example_DMRG_hubbard_infinite_S_xi_scaling(Vpm):
    ROOTDIR = f'../../data/iDMRG/Vpm{Vpm:.3f}cc/'
    os.makedirs(ROOTDIR, exist_ok=True)
    model_params = dict(
        t=1.0, U=0.1, Vpp=0.8, Vpm=Vpm, mu=0.0, hx=0.0, hz=-1e-10,
        bc_MPS="infinite",
        L=4, cons_N="N", cons_Sz="None",
    )
    M = HubbardSpinDepV(model_params)
    # psi, _  = io.load_mps_with_metadata("../../data/iDMRG/v2fwd/t_1.00_U_0.10_Vpp_0.800_consNf0_25_chimax_100/Vpm_3.8000_chimax_100.h5")
    # psi, _  = io.load_mps_with_metadata(f"./Vpm3.8chi80.h5")
    # psi, _ = io.load_mps_with_metadata(os.path.join(ROOTDIR, f"Vpm_{Vpm:.3f}_chi1000.h5"))
    # psi, = io.load_mps_with_metadata('../../data/iDMRG/scanChi500/Vpm_4.800_chi500.h5') #initial ferro state
    # psi, _ = io.load_mps_with_metadata('Vpm_6.000_chi_100.h5')
    # psi, _ = io.load_mps_with_metadata(f'../../data/iDMRG/rev/scanChi200/Vpm_{Vpm:.3f}_chi200.h5')
    psi, _ = io.load_mps_with_metadata(os.path.join(ROOTDIR, f"Vpm_{Vpm:.3f}_chi500.h5"))
    print("loaded intital psi, starting ...........", flush=True)
    psi.canonical_form_infinite2()
    dmrg_params = {
        'start_env': 10,
        # 'mixer': False,
        'mixer' : True,
        'mixer_params': {'amplitude': 1e-2, 'decay': 1.2, 'disable_after': 100},
        # 'trunc_params': {'chi_max': 1098, 'svd_min': 1.0e-10},
        'trunc_params': {'chi_max': 200, 'svd_min': 1.0e-12},
        'max_E_err': 1.0e-8, #was -7
        'max_S_err': 1.0e-6, #was -7
        'update_env': 0,
    }

    # chi_list = np.arange(7, 31, 2)
    # chi_list = np.arange(80, 501, 20)
    # chi_list = np.arange(200, 501, 20)
    chi_list = np.arange(550, 1001, 50)
    # chi_list = np.arange(1100, 2001, 100)
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
        xi_list.append(psi.correlation_length2())

        print(chi, time.time() - t0, s_list[-1], xi_list[-1], flush=True)
        # tenpy.tools.optimization.optimize(3)  # quite some speedup for small chi
        
        # if chi % 50 == 0:
        if True:
            results = {
                "diagnostics" : eng.sweep_stats, 
                "model_params" : model_params, 
                "dmrg_params" : dmrg_params,
            }
            io.save_mps_with_metadata(os.path.join(ROOTDIR, f"Vpm_{Vpm:.3f}_chi{chi}.h5"), psi, results)

        print('SETTING NEW BOND DIMENSION')

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
    # s_list, xi_list = example_DMRG_hubbard_infinite_S_xi_scaling(Vpm=3.8) #Deep in para phase 
    Vpm_list = (4.30, 4.35, 4.4, 4.5, 4.6)
    idx = int(os.getenv("SLURM_ARRAY_TASK_ID", 1))
    Vpm = Vpm_list[idx]
    print("SLURM ARRAY TASK : ", idx, " Vpm = ", Vpm, flush=True)
    s_list, xi_list = example_DMRG_hubbard_infinite_S_xi_scaling(Vpm=Vpm)
    # fit_plot_central_charge(s_list, xi_list, 'central_charge_para80_300.pdf')
    fit_plot_central_charge(s_list, xi_list, f'central_charge_fm_Vpm_{Vpm:.3f}_cc.pdf')
