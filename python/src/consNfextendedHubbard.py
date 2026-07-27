##############################################################################
# 1D Hubbard model with spin-dependent NN repulsion — Kun Yang (2004)
# iDMRG via TeNPy
#
# Hamiltonian (Eq. 1 of cond-mat/0401149, spin-dependent V branch):
#
#   H = -t  Σ_{i,σ} (c†_{i,σ} c_{i+1,σ} + h.c.)
#       + U  Σ_i  n_{i↑} n_{i↓}
#       + V^{++} Σ_i  (n_{i↑}n_{i+1,↑} + n_{i↓}n_{i+1,↓})   same-spin
#       + V^{+-} Σ_i  (n_{i↑}n_{i+1,↓} + n_{i↓}n_{i+1,↑})   opposite-spin
#
# FM instability condition (paper, coefficient a of Eq. 7):
#   a ∝ Σ_j j^2 (V^{+-}_j - V^{++}_j)  > 0
#
# i.e.  V^{+-} > V^{++}  makes a > 0, which is required for the second-order
# FM transition (rather than a first-order one).  The spin stiffness v_{Ns}
# then softens to zero at the critical point.
#
# Usage:
#   pip install physics-tenpy
#   python hubbard_spindep_V_tenpy_idmrg.py                  # single run
#   python hubbard_spindep_V_tenpy_idmrg.py --mode scan      # scan ΔV
#   python hubbard_spindep_V_tenpy_idmrg.py --mode finite    # finite DMRG
##############################################################################

import argparse
import numpy as np
import tenpy
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinHalfFermionSite
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg as tenpy_dmrg
from tenpy.algorithms.vumps import TwoSiteVUMPSEngine
import sys
sys.path.append('..')
import utils.io as io
import os 

# print(f"TeNPy version: {tenpy.__version__}")


# ─── Model definition ─────────────────────────────────────────────────────────
class HubbardSpinDepV(CouplingMPOModel):
    """
    1D Hubbard model with spin-dependent nearest-neighbor repulsion.

    Model parameters:
        t     : float — NN hopping (default 1.0)
        U     : float — on-site Hubbard repulsion 
        Vpp   : float — V^{++} = V^{--}, same-spin NN repulsion 
        Vpm   : float — V^{+-} = V^{-+}, opposite-spin NN repulsion
                        FM condition: Vpm > Vpp
        hx    : float - hx
        hz    : float - hz
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


# ─── iDMRG ───────────────────────────────────────────────────────────────────
def run_idmrg(t=1.0, U=4.0, Vpp=0.5, Vpm=1.5, chi_max=100, n_sweeps=12, hx=0.0, hz=0.0, max_err=1e-5, L=4,
              verbose=True, psi_init=None, diagnostics=False):
    """
    Run iDMRG for HubbardSpinDepV in the thermodynamic limit.
    Unit cell L=2 (minimum for a non-trivial iMPS at half filling).
    """
    if verbose:
        print("=" * 65)
        print(f"  t={t}  U={U}  V^{{++}}={Vpp}  V^{{+-}}={Vpm}")
        print(f"  ΔV = V^{{+-}} - V^{{++}} = {Vpm-Vpp:.3f}  (>0 drives FM)")
        print("=" * 65)

    
    chi_list = {0:chi_max//2, 21: int(3*chi_max/4), 51:chi_max}
    dmrg_params = {
        "trunc_params": {
            "chi_max": chi_max,
            "svd_min": 1e-11,
            "trunc_cut": 1e-8,
        },
        "chi_list":chi_list,
        "mixer": "DensityMatrixMixer",
        "mixer_params": {
            "amplitude": 1e-5,
            "decay": 1.5,
            "disable_after": 30,
        },
        # "N_sweeps_check": 1,
        "min_sweeps": 30,
        "max_sweeps": n_sweeps,
        "update_env": 4,
        # "start_env": 10,
        'max_E_err': max_err, #precision in energy 
        'max_S_err': max_err, #precision in entropy
    }
    model_params = dict(
        t=t, U=U, Vpp=Vpp, Vpm=Vpm, mu=0.0, hx=hx, hz=hz,
        bc_MPS="infinite",
        L=L, cons_N="N", cons_Sz="None",
    )
    model = HubbardSpinDepV(model_params)
    if not psi_init:
        # Quarter-filling, Sz=N/2 initial state
        if L == 2 :
            psi = MPS.from_product_state(model.lat.mps_sites(), ["up", "empty"] , bc=model.lat.bc_MPS)
        elif L == 4:
            psi = MPS.from_product_state(model.lat.mps_sites(), ["up", "empty", "up", "empty"] , bc=model.lat.bc_MPS)
    else : 
        psi = psi_init
        chi_init = max(psi.chi)
        if chi_init > chi_max // 2: 
            dmrg_params["chi_list"] = None
            dmrg_params["mixer"] = False
            dmrg_params["start_env"] = 0


    eng = tenpy_dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    E, psi = eng.run()   # E is energy per site for iDMRG
    total_sweeps = eng.sweeps
    converged = eng.is_converged()

    if not converged and total_sweeps >= eng.options['max_sweeps']:
        print(f"Halted at maximum sweeps limit ({total_sweeps}) without fully converging.")

    results = {
        "model_params" : model_params, 
        "dmrg_params"  : dmrg_params,
        "converged"    : converged,
        }
    if verbose:
        report                  = _report_idmrg(E, psi)
        results.update(report)

    if diagnostics:
        results["diagnostics"] = {
            "sweep_stats": eng.sweep_stats,
            # "update_stats": eng.update_stats, # we don't want this because it takes up a shyat-ton of memory
            "age" : eng.update_stats["age"][::10] # just some additional compression
        }

    return E, psi, model, results


def _report_idmrg(E, psi):
    Sz    = psi.expectation_value("Sz")
    Nup   = psi.expectation_value("Nu")
    Ndown = psi.expectation_value("Nd")
    Ntot  = psi.expectation_value("Ntot")

    print(f"\nGround state energy per site: E/N = {E:.10f}")
    print(f"\nUnit-cell observables:")
    print(f"  {'site':>4}  {'<Sz>':>10}  {'<Nup>':>10}  {'<Ndn>':>10}  {'<Ntot>':>10}")
    print("  " + "-" * 50)
    for i in range(len(Sz)):
        print(f"  {i:4d}  {Sz[i]:+10.6f}  {Nup[i]:10.6f}"
              f"  {Ndown[i]:10.6f}  {Ntot[i]:10.6f}")

    mean_Sz   = float(np.mean(Sz))
    mean_fill = float(np.mean(Ntot))
    print(f"\n  Mean <Sz>     = {mean_Sz:+.6f}")
    print(f"  Mean filling  = {mean_fill:.6f}  (should be 1.0 at half filling)")

    # Spin-spin correlations and susceptibility
    corr = psi.correlation_function("Sz", "Sz")
    chi_q0 = float(np.sum(corr[0, :]))
    print(f"\n  Spin susceptibility (q=0 sum): χ ≈ {chi_q0:.4f}")

    results = {
        "mean_Sz" : mean_Sz, 
        "mean_fill" : mean_fill,
        "chi_q0" : chi_q0,
    }
    # Spin structure factor S(q)
    # L = len(Sz)
    # print("\n  Spin structure factor S(q) [from unit-cell correlations]:")
    # for q_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
    #     q = q_frac * np.pi
    #     Sq = 0.0
    #     for i in range(L):
    #         for j in range(corr.shape[1]):
    #             Sq += np.cos(q * (i - j)) * corr[i, j]
    #     Sq /= L
    # print(f"    q/π = {q_frac:.2f}:  S(q) = {Sq:+.6f}")
    # if abs(chi_q0) > 5.0:
    #     print("\n  >>> Large χ(q=0) — system is near or in the FM phase!")
    # else:
    #     print("\n  >>> Paramagnetic ground state at these parameters.")

    return results



# ─── Finite DMRG ─────────────────────────────────────────────────────────────
def run_finite_dmrg(N=20, t=1.0, U=4.0, Vpp=0.5, Vpm=1.5,
                    chi_max=400, n_sweeps=10):
    """
    Finite-chain open-boundary DMRG. Useful for benchmarking and for
    measuring real-space correlations directly.
    """
    print("=" * 65)
    print("Finite DMRG: Hubbard + spin-dependent NN repulsion")
    print(f"  N={N}  t={t}  U={U}  V^{{++}}={Vpp}  V^{{+-}}={Vpm}")
    print("=" * 65)

    model_params = dict(
        t=t, U=U, Vpp=Vpp, Vpm=Vpm, mu=0.0,
        bc_MPS="finite",
        L=N,
        cons_N="None",
        cons_Sz="None",
    )
    model = HubbardSpinDepV(model_params)

    state = ["up" if i % 2 == 0 else "down" for i in range(N)]
    psi = MPS.from_lat_product_state(model.lat, state)

    dmrg_params = {
        "trunc_params": {"chi_max": chi_max, "svd_min": 1e-11},
        "mixer": "DensityMatrixMixer",
        "mixer_params": {"amplitude": 1e-4, "decay": 1.5, "disable_after": 40},
        "N_sweeps_check": 1,
        "min_sweeps": 4,
        "max_sweeps": n_sweeps,
        "norm_tol": 1e-8,
    }

    eng = tenpy_dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    E, psi = eng.run()

    Sz    = psi.expectation_value("Sz")
    Ntot  = psi.expectation_value("Ntot")
    SzSz  = psi.correlation_function("Sz", "Sz")

    print(f"\nTotal energy:       E   = {E:.10f}")
    print(f"Energy per site:    E/N = {E/N:.10f}")
    print(f"Total <Ntot>:       {sum(Ntot):.4f}  (expected {N})")
    print(f"Total |<Sz>|:       {abs(sum(Sz)):.6f}")

    print("\nSite-resolved <Sz>:")
    for i, sz in enumerate(Sz):
        bar = "█" * int(abs(sz) * 40)
        print(f"  site {i:3d}: {sz:+.6f}  {bar}")

    # Spin structure factor
    print("\nSpin structure factor S(q):")
    for q_frac in np.arange(0, 1.01, 0.125):
        q = q_frac * np.pi
        Sq = sum(np.cos(q*(i-j)) * SzSz[i,j]
                 for i in range(N) for j in range(N)) / N
        print(f"  q/π = {q_frac:.3f}:  S(q) = {Sq:+.6f}")

    return E, psi


# ─── Scan ΔV to locate FM transition ─────────────────────────────────────────
def scan_deltaV(t=1.0, U=0.1, Vpp=0.8,
                dV_values=None, chi_max=100, n_sweeps=500, max_err=1e-5, saveflag=False, diagnostics=True, ROOTDIR='../../data/iDMRG/'):

    if dV_values is None:
        dV_values = np.arange(2.9, 4.0, 0.1)[::-1]

    print("\n" + "=" * 70)
    print("Scanning ΔV = V^{+-} - V^{++} to map PM → FM transition")
    print(f"  t={t}  U={U}  V^{{++}}={Vpp}")
    print("=" * 70)
    print(f"  {'ΔV':>6}  {'V^{+-}':>8}  {'E/N':>14}  {'χ(q=0)':>12}  {'S(q=0)':>10}")
    print("  " + "-" * 58, flush=True)

    chi_list = {0:chi_max//2, 21: int(3*chi_max/4), 51:chi_max}
    dmrg_params = {
        "trunc_params": {
            "chi_max": chi_max,
            "svd_min": 1e-11,
            "trunc_cut": 1e-8,
        },
        "chi_list":chi_list,
        "mixer": "DensityMatrixMixer",
        "mixer_params": {
            "amplitude": 1e-5,
            "decay": 1.2,
            "disable_after": 30,
        },
        # "N_sweeps_check": 1,
        "min_sweeps": 30,
        "max_sweeps": n_sweeps,
        "update_env": 10,
        # "start_env": 10,
        'max_E_err': max_err, #precision in energy 
        'max_S_err': max_err, #precision in entropy
    }
    psi = None #first run keep psi None to init engine etc
    converged = None
    if saveflag:
        JOBDIR = os.path.join(ROOTDIR, f"t_{t:.2f}_U_{U:.2f}_Vpp_{Vpp:.3f}_consNf0_25_chimax_{chi_max}")
        os.makedirs(JOBDIR, exist_ok=True)

    for dV in dV_values:
        Vpm = Vpp + dV
        model_params = dict(
            t=t, U=U, Vpp=Vpp, Vpm=Vpm, mu=0.0,
            bc_MPS="infinite", L=2, cons_N="N", cons_Sz="None",
        )
        model = HubbardSpinDepV(model_params)
        if psi is None:
            psi = MPS.from_product_state(model.lat.mps_sites(), ["up", "empty"] , bc=model.lat.bc_MPS)
            eng = tenpy_dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
        else:
            eng.init_env(model=model)
        E, psi = eng.run()
        total_sweeps = eng.sweeps
        converged = eng.is_converged()
        if not converged and total_sweeps >= eng.options['max_sweeps']:
            print(f"Halted at maximum sweeps limit ({total_sweeps}) without fully converging.")
        else: 
            print(r"Congratulations, converged!")

        corr  = psi.correlation_function("Sz", "Sz")
        chi   = float(np.sum(corr[0, :]))
        Sz    = psi.expectation_value("Sz")
        L     = len(Sz)
        Sq0   = float(np.sum(corr)) / L
        xi    = psi.correlation_length()
        Ntot  = psi.expectation_value("Ntot")

        results = {
            "Sz"           : Sz, 
            "Ntot"         : Ntot,
            "E"            : E, 
            "mean_Sz"      : float(np.mean(Sz)), 
            "mean_fill"    : float(np.mean(Ntot)),
            "converged"    : converged, 
            "xi"           : xi, 
            "model_params" : model_params, 
            "dmrg_params"  : dmrg_params,
        }
        if diagnostics:
            results["diagnostics"] = {
                "sweep_stats": eng.sweep_stats,
                # "update_stats": eng.update_stats, # we don't want this because it takes up a shyat-ton of memory
                "age" : eng.update_stats["age"][::10] # just some additional compression
            }

        if saveflag:
            savefilename = os.path.join(JOBDIR, f"Vpm_{Vpm:.4f}_chimax_{chi_max}.h5")
            io.save_mps_with_metadata(savefilename, psi, results)


        # if converged:
        #     dmrg_params['start_env'] = 0 #the environment is already built, no need to waste time rebuilding it # don't risk it, maybe set like 2 in the future
        # else: 
        #     dmrg_params['start_env'] = 10
        print(f"  {dV:6.3f}  {Vpm:8.3f}  {E:14.8f}  {chi:12.4f}  {Sq0:10.4f}, {xi:10.4f}")



# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="iDMRG for Hubbard + spin-dep V model (Kun Yang 2004)")
    parser.add_argument("--mode",   choices=["idmrg", "finite", "scan"],
                        default="idmrg")
    parser.add_argument("--t",      type=float, default=1.0)
    parser.add_argument("--U",      type=float, default=1.0)
    parser.add_argument("--Vpp",    type=float, default=1.0,
                        help="V^{++}: same-spin NN repulsion")
    parser.add_argument("--Vpm",    type=float, default=3.5,
                        help="V^{+-}: opposite-spin NN repulsion (FM: Vpm > Vpp)")
    parser.add_argument("--N",      type=int,   default=20,
                        help="chain length (finite mode only)")
    parser.add_argument("--chi",    type=int,   default=300,
                        help="max bond dimension")
    parser.add_argument("--sweeps", type=int,   default=12)
    parser.add_argument("--max_err", type=float, default=1e-5)
    args = parser.parse_args()

    if args.mode == "idmrg":
        run_idmrg(t=args.t, U=args.U, Vpp=args.Vpp, Vpm=args.Vpm,
                  chi_max=args.chi, n_sweeps=args.sweeps, max_err=args.max_err)

    elif args.mode == "finite":
        run_finite_dmrg(N=args.N, t=args.t, U=args.U, Vpp=args.Vpp, Vpm=args.Vpm,
                        chi_max=args.chi, n_sweeps=args.sweeps)

    elif args.mode == "scan":
        scan_deltaV(t=args.t, U=args.U, Vpp=args.Vpp,
                    chi_max=args.chi, n_sweeps=args.sweeps, max_err=args.max_err)