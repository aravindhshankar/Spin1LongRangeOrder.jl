module Hamiltonians

using ITensors, ITensorMPS

export build_fm_hamiltonian, build_prelim_hamiltonian, build_pert_hamiltonian, build_g1g2_hamiltonian

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

function build_g1g2_hamiltonian(sites::Vector, J::Real, g1::Real, g2::Real)
  N = length(sites)
  os = OpSum()
  for j = 1:N-1
    os += J, "Sz", j, "Sz", j + 1
    os += 0.5 * J, "S+", j, "S-", j + 1
    os += 0.5 * J, "S-", j, "S+", j + 1
    os += g1, "Sz2", j
    os += g2, "Sz2", j, "Sz2", j + 1
  end
  return MPO(os, sites)
end
build_g1g2_hamiltonian(sites::Vector; J::Real, g1::Real, g2::Real) = build_g1g2_hamiltonian(sites, J, g1, g2)

"""
    build_g1g2_hamiltonian(sites::Vector, J::Real, g1::Real, g2::Real, hx::Real, hz::Real)

Build g1g2 Hamiltonian with Sz^2 and pair Sz^2 terms plus transverse (hx) and longitudinal (hz) fields.
"""
function build_g1g2_hamiltonian(sites::Vector, J::Real, g1::Real, g2::Real, hx::Real, hz::Real)
  N = length(sites)
  os = OpSum()
  for j = 1:N-1
    os += J, "Sz", j, "Sz", j + 1
    os += 0.5 * J, "S+", j, "S-", j + 1
    os += 0.5 * J, "S-", j, "S+", j + 1
    os += g1, "Sz2", j
    os += g2, "Sz2", j, "Sz2", j + 1
  end
  for j = 1:N
    os += hx, "Sx", j
    os += hz, "Sz", j
  end
  return MPO(os, sites)
end
build_g1g2_hamiltonian(sites::Vector; J::Real, g1::Real, g2::Real, hx::Real, hz::Real) = build_g1g2_hamiltonian(sites, J, g1, g2, hx, hz)

"""
Build g1g2 Hamiltonian with boundary operator term added at the ends.
"""
function build_g1g2_hamiltonian(sites::Vector, J::Real, g1::Real, g2::Real, boundary_op::String, boundary_h::Real)
  N = length(sites)
  os = OpSum()
  for j = 1:N-1
    os += J, "Sz", j, "Sz", j + 1
    os += 0.5 * J, "S+", j, "S-", j + 1
    os += 0.5 * J, "S-", j, "S+", j + 1
    os += g1, "Sz2", j
    os += g2, "Sz2", j, "Sz2", j + 1
  end
  os += boundary_h, boundary_op, 1
  os += boundary_h, boundary_op, N
  return MPO(os, sites)
end
build_g1g2_hamiltonian(sites::Vector; J::Real, g1::Real, g2::Real, boundary_op::String, boundary_h::Real) = build_g1g2_hamiltonian(sites, J, g1, g2, boundary_op, boundary_h)



end # module Hamiltonians
