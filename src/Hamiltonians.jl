module Hamiltonians

using ITensors, ITensorMPS

export build_fm_hamiltonian, build_prelim_hamiltonian, build_pert_hamiltonian

"""
    build_fm_hamiltonian(sites::Vector, J::Real)

Build ferromagnetic Hamiltonian with nearest-neighbor interactions.
"""
function build_fm_hamiltonian(sites::Vector, J::Real)
    N = length(sites)
    os = OpSum()
    for j = 1:N-1
        os += J, "Sz", j, "Sz", j + 1
        os += 0.5 * J, "S+", j, "S-", j + 1
        os += 0.5 * J, "S-", j, "S+", j + 1
    end
    return MPO(os, sites)
end

"""
    build_prelim_hamiltonian(sites::Vector, J::Real)

Build preliminary Hamiltonian with single-site Sz term.
"""
function build_prelim_hamiltonian(sites::Vector, J::Real)
    N = length(sites)
    os = OpSum()
    for j = 1:N-1
        os += J, "Sz", j, "Sz", j + 1
        os += 0.5 * J, "S+", j, "S-", j + 1
        os += 0.5 * J, "S-", j, "S+", j + 1
        os += 1, "Sz", j
    end
    return MPO(os, sites)
end
"""
    build_pert_hamiltonian(sites::Vector, J::Real, g::Real)

Build perturbed Hamiltonian with Sz^2 term.
"""
function build_pert_hamiltonian(sites::Vector, J::Real, g::Real)
    N = length(sites)
    os = OpSum()
    for j = 1:N-1
        os += J, "Sz", j, "Sz", j + 1
        os += 0.5 * J, "S+", j, "S-", j + 1
        os += 0.5 * J, "S-", j, "S+", j + 1
        os += g, "Sz2", j
    end
    return MPO(os, sites)
end

end