using Spin1LongRangeOrder
using Spin1LongRangeOrder.Hamiltonians
using ITensorMPS
using Printf


function runDMRGaction(hamparams)
  N = 10
  # hamparams = (J=1, g1=0.5, g2=-0.5)
  sites = siteinds("S=1", N)
  Ham = build_g1g2_hamiltonian(sites; hamparams...)
  println("Built hamiltonian")

  # psi_0 = randomMPS(sites, linkdims=10)
  # instate = ["Up" for n in 1:N]
  instate = ["X+" for _ in 1:N]
  psi_0 = random_mps(sites, instate; linkdims=10)

  nsweeps = 6
  maxdim = [10, 20]
  mindim = [1, 1]
  eigsolve_krylovdim = 3
  cutoff = [1E-12]
  noise = [1E-6, 0]
  obsparams = (energy_tol=1E-5, minsweeps=5, energy_type=Float64)
  observer = DMRGObserver(; obsparams...)

  energy, psi = dmrg(Ham, psi_0; nsweeps, maxdim, cutoff, mindim=mindim,
    eigsolve_krylovdim=eigsolve_krylovdim, observer, noise)

  fmt(x) = @sprintf("%.2f", x)
  savepath = "data/N10/"
  mkpath(savepath)
  parts = [String(k) * fmt(v) for (k, v) in pairs(hamparams)]
  filename = savepath * join(parts, "") * ".h5"
  save_simulation(filename, psi, Dict(pairs(hamparams)))
  println("saved")
end

let
  gval = 0.6
  hamparams = (J=1, g1=gval, g2=-0.5)
  runDMRGaction(hamparams)
end


