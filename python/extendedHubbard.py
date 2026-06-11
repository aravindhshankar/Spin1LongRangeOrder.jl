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

print(f"TeNPy version: {tenpy.__version__}")


# ─── Model definition ─────────────────────────────────────────────────────────
class HubbardSpinDepV(CouplingMPOModel):
    """
    1D Hubbard model with spin-dependent nearest-neighbor repulsion.

    Model parameters:
        t     : float — NN hopping (default 1.0)
        U     : float — on-site Hubbard repulsion (default 4.0)
        Vpp   : float — V^{++} = V^{--}, same-spin NN repulsion (default 0.5)
        Vpm   : float — V^{+-} = V^{-+}, opposite-spin NN repulsion (default 1.5)
                        FM condition: Vpm > Vpp
        mu    : float — chemical potential (default 0.0, half filling)
        bc_MPS    : 'infinite' or 'finite'
        L         : unit cell / chain length
    """

    def init_sites(self, model_params):
        cons_N = model_params.get("cons_N", "None")
        cons_Sz = model_params.get("cons_Sz", "None")
        return SpinHalfFermionSite(cons_N = cons_N, cons_Sz = cons_Sz)

    def init_terms(self, model_params):
        t   = model_params.get("t",   1.0)
        U   = model_params.get("U",   4.0)
        Vpp = model_params.get("Vpp", 0.5)   # V^{++}: same spin
        Vpm = model_params.get("Vpm", 1.5)   # V^{+-}: opposite spin
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
        self.add_coupling(Vpm, 0, "Nu",   0, "Nd", 1)
        self.add_coupling(Vpm, 0, "Nd", 0, "Nu",   1)

        # Chemical potential
        if abs(mu) > 1e-12:
            self.add_onsite(-mu, 0, "Ntot")


# ─── iDMRG ───────────────────────────────────────────────────────────────────
def run_idmrg(t=1.0, U=4.0, Vpp=0.5, Vpm=1.5, chi_max=300, n_sweeps=12,
              verbose=True):
    """
    Run iDMRG for HubbardSpinDepV in the thermodynamic limit.
    Unit cell L=2 (minimum for a non-trivial iMPS at half filling).
    """
    if verbose:
        print("=" * 65)
        print("iDMRG: Hubbard + spin-dependent NN repulsion (Kun Yang 2004)")
        print(f"  t={t}  U={U}  V^{{++}}={Vpp}  V^{{+-}}={Vpm}")
        print(f"  ΔV = V^{{+-}} - V^{{++}} = {Vpm-Vpp:.3f}  (>0 drives FM)")
        print("=" * 65)

    model_params = dict(
        t=t, U=U, Vpp=Vpp, Vpm=Vpm, mu=0.0,
        bc_MPS="infinite",
        L=2,
    )
    model = HubbardSpinDepV(model_params)

    # Half-filling, Sz=0 initial state
    psi = MPS.from_product_state(model.lat.mps_sites(), ["up", "down"] , bc=model.lat.bc_MPS)

    dmrg_params = {
        "trunc_params": {
            "chi_max": chi_max,
            "svd_min": 1e-11,
        },
        "mixer": "DensityMatrixMixer",
        "mixer_params": {
            "amplitude": 1e-5,
            "decay": 1.5,
            "disable_after": 60,
        },
        "N_sweeps_check": 1,
        "min_sweeps": 4,
        "max_sweeps": n_sweeps,
        "norm_tol": 1e-6,
        "update_env": 10,
        "start_env": 10,
    }

    eng = tenpy_dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    E, psi = eng.run()   # E is energy per site for iDMRG

    if verbose:
        _report_idmrg(E, psi)

    return E, psi, model


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

    # Spin structure factor S(q)
    L = len(Sz)
    print("\n  Spin structure factor S(q) [from unit-cell correlations]:")
    for q_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        q = q_frac * np.pi
        Sq = 0.0
        for i in range(L):
            for j in range(corr.shape[1]):
                Sq += np.cos(q * (i - j)) * corr[i, j]
        Sq /= L
        print(f"    q/π = {q_frac:.2f}:  S(q) = {Sq:+.6f}")

    if abs(chi_q0) > 5.0:
        print("\n  >>> Large χ(q=0) — system is near or in the FM phase!")
    else:
        print("\n  >>> Paramagnetic ground state at these parameters.")


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
def scan_deltaV(t=1.0, U=4.0, Vpp=0.5,
                dV_values=None, chi_max=150, n_sweeps=8):
    """
    Fix V^{++} and scan V^{+-} - V^{++} from 0 upward.
    The FM transition is signalled by:
      - diverging χ(q=0)
      - growing S(q=0) / spin structure factor at q=0
    """
    if dV_values is None:
        dV_values = np.arange(0.0, 2.25, 0.25)

    print("\n" + "=" * 70)
    print("Scanning ΔV = V^{+-} - V^{++} to map PM → FM transition")
    print(f"  t={t}  U={U}  V^{{++}}={Vpp}")
    print("=" * 70)
    print(f"  {'ΔV':>6}  {'V^{+-}':>8}  {'E/N':>14}  {'χ(q=0)':>12}  {'S(q=0)':>10}")
    print("  " + "-" * 58)

    for dV in dV_values:
        Vpm = Vpp + dV
        model_params = dict(
            t=t, U=U, Vpp=Vpp, Vpm=Vpm, mu=0.0,
            bc_MPS="infinite", L=2, conserve="best",
        )
        model = HubbardSpinDepV(model_params)
        psi = MPS.from_lat_product_state(model.lat, ["up", "down"])

        dmrg_params = {
            "trunc_params": {"chi_max": chi_max, "svd_min": 1e-10},
            "mixer": "DensityMatrixMixer",
            "mixer_params": {"amplitude": 1e-5, "decay": 1.5, "disable_after": 30},
            "N_sweeps_check": 1, "min_sweeps": 3, "max_sweeps": n_sweeps,
            "norm_tol": 1e-5, "update_env": 5, "start_env": 5,
        }
        eng = tenpy_dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
        E, psi = eng.run()

        corr  = psi.correlation_function("Sz", "Sz")
        chi   = float(np.sum(corr[0, :]))
        Sz    = psi.expectation_value("Sz")
        L     = len(Sz)
        Sq0   = float(np.sum(corr)) / L

        print(f"  {dV:6.3f}  {Vpm:8.3f}  {E:14.8f}  {chi:12.4f}  {Sq0:10.4f}")


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
    args = parser.parse_args()

    if args.mode == "idmrg":
        run_idmrg(t=args.t, U=args.U, Vpp=args.Vpp, Vpm=args.Vpm,
                  chi_max=args.chi, n_sweeps=args.sweeps)

    elif args.mode == "finite":
        run_finite_dmrg(N=args.N, t=args.t, U=args.U, Vpp=args.Vpp, Vpm=args.Vpm,
                        chi_max=args.chi, n_sweeps=args.sweeps)

    elif args.mode == "scan":
        scan_deltaV(t=args.t, U=args.U, Vpp=args.Vpp,
                    chi_max=args.chi, n_sweeps=args.sweeps)