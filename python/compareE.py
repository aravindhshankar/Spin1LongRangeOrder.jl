import numpy as np

from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinSite
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS


class Spin1Chain(CouplingMPOModel):
    """
    Spin-1 chain with Hamiltonian

        H = -J Σ_i S_i · S_{i+1}
            + g1 Σ_i (S_i^z)^2
            + g2 Σ_i (S_i^z)^2 (S_{i+1}^z)^2

    Infinite boundary conditions (iMPS / iDMRG).
    """

    def init_sites(self, model_params):
        conserve = model_params.get("conserve", None)
        site = SpinSite(S=model_params["S"], conserve=conserve)
        return site

    def init_terms(self, model_params):
        J = model_params.get("J", 1.0)
        g1 = model_params.get("g1", 0.0)
        g2 = model_params.get("g2", 0.0)

        # --- nearest-neighbor Heisenberg coupling ---
        # H = -J (SxSx + SySy + SzSz)

        for u1, u2, dx in self.lat.pairs["nearest_neighbors"]:
            self.add_coupling(-0.5*J, u1, "Sp", u2, "Sm", dx, plus_hc=True)
            self.add_coupling(-J, u1, "Sz", u2, "Sz", dx)

        # --- single-ion anisotropy g1 (Sz)^2 ---
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(g1, u, "Sz Sz")

        # --- biquadratic anisotropy g2 (Sz_i)^2 (Sz_j)^2 ---
        for u1, u2, dx in self.lat.pairs["nearest_neighbors"]:
            self.add_coupling(g2, u1, "Sz Sz", u2, "Sz Sz", dx)
        # for u1, u2, dx in self.lat.pairs["nearest_neighbors"]:
        #     self.add_multi_coupling(
        #         g2,
        #         [
        #             (u1, "Sz Sz", 0),
        #             (u2, "Sz Sz", dx),
        #         ],
        #     )


# -------------------------------------------------------------------
# Model parameters
# -------------------------------------------------------------------

model_params = {
    "L": 2,                 # unit cell size for iMPS
    "S": 1,
    "J": 1.0,
    "g1": 0.5,
    "g2": -0.2,
    "bc_MPS": "infinite",  # infinite boundary conditions
    "conserve": None,
}

model = Spin1Chain(model_params)

# -------------------------------------------------------------------
# Initial iMPS product state
# -------------------------------------------------------------------

product_state = ["up", "down"]  # repeating unit cell

psi = MPS.from_product_state(
    model.lat.mps_sites(),
    product_state,
    bc="infinite",
    unit_cell_width=len(product_state)
)

# -------------------------------------------------------------------
# iDMRG parameters
# -------------------------------------------------------------------

dmrg_params = {
    "mixer": True,
    "trunc_params": {
        "chi_max": 100,
        "svd_min": 1e-10,
    },
    "max_E_err": 1e-8,
    "max_S_err": 1e-6,
    "max_sweeps": 30,
    "verbose": 1,
}

# -------------------------------------------------------------------
# Run infinite DMRG
# -------------------------------------------------------------------

info = dmrg.run(psi, model, dmrg_params)

print("\n=== iDMRG finished ===")
print("Ground state energy density =", info["E"])
print(info.keys())

# -------------------------------------------------------------------
# Example observables
# -------------------------------------------------------------------

Sz = psi.expectation_value("Sz")
Sz2 = psi.expectation_value("Sz Sz")

print("\n<Sz> per site:")
print(Sz)

print("\n<(Sz)^2> per site:")
print(Sz2)

# nearest-neighbor correlations
corr = psi.correlation_function("Sz", "Sz", [0], [1])

print("\n<Sz_i Sz_{i+1}>:")
print(corr)
