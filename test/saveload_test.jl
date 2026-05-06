using Spin1LongRangeOrder
using Spin1LongRangeOrder.Hamiltonians
using ITensorMPS

let
  N = 10
  hamparams = (J=1, g1=1, g2=0.5)
  sites = siteinds("S=1", N)
  Ham = build_g1g2_hamiltonian(sites; hamparams...)
  println("Built hamiltonian")

  psi_0 = randomMPS(sites, linkdims=10)

  nsweeps = 5
  maxdim = [10, 20]
  mindim = [1, 1]
  eigsolve_krylovdim = 3
  cutoff = [1E-12]
  noise = [1E-6, 0]

  energy, psi = dmrg(Ham, psi_0; nsweeps, maxdim, cutoff, mindim=mindim,
    eigsolve_krylovdim=eigsolve_krylovdim)

  save_simulation("test/saveload_test.h5", psi, Dict(pairs(hamparams)))
  println("saved")

  load_psi, load_params = load_simulation("test/saveload_test.h5", Val(:params))
  println("loaded")

end




