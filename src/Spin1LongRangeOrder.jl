module Spin1LongRangeOrder

using ITensors, ITensorMPS
using Plots
gr()

let
  N = 100
  sites = siteinds("S=1", N)
  h = 0.0
  J = -1.0

  os = OpSum()
  for j = 1:N-1
    os += J, "Sz", j, "Sz", j + 1
    os += 0.5 * J, "S+", j, "S-", j + 1
    os += 0.5 * J, "S-", j, "S+", j + 1
    # os += 0.01, "Sz", j
  end
  H = MPO(os, sites)

  os_prelim = OpSum()
  for j = 1:N-1
    os_prelim += J, "Sz", j, "Sz", j + 1
    os_prelim += 0.5 * J, "S+", j, "S-", j + 1
    os_prelim += 0.5 * J, "S-", j, "S+", j + 1
    os_prelim += 1, "Sz", j
  end
  H_prelim = MPO(os_prelim, sites)

  nsweeps_p = 5
  maxdim_p = [5, 10, 20, 50, 100] # gradually increase states kept
  # mindim = [1, 1, 5, 10, 20, 50]
  mindim_p = [1]
  eigsolve_krylovdim_p = 3 #default 3
  cutoff_p = [1E-12] # desired truncation error
  noise_p = [1e-6]

  psi0 = random_mps(sites; linkdims=2)

  energy, psi_p = dmrg(H_prelim, psi0; nsweeps=nsweeps_p, maxdim=maxdim_p,
    cutoff=cutoff_p, mindim=mindim_p,
    eigsolve_krylovdim=eigsolve_krylovdim_p, noise=noise_p)

  nsweeps = 2
  maxdim = [100]
  mindim = [1, 1]
  eigsolve_krylovdim = 3
  cutoff = [1E-12]
  noise = [0]
  energy, psi = dmrg(H, psi_p; nsweeps, maxdim, cutoff, mindim=mindim,
    eigsolve_krylovdim=eigsolve_krylovdim)
  println("Energy = ", energy)
  zzcorr = correlation_matrix(psi, "Sz", "Sz")

  p = plot(abs.(zzcorr[Int64(N // 2), :]), ms=5, lw=2)
  # plot!(xscale=:identity, yscale=:log10, minorgrid=true)
  plot!(xscale=:identity, yscale=:identity, minorgrid=true)
  # plot!(xscale=:log10, yscale=:log10, minorgrid=true)
  savefig(p, "figure.png")
  println("saved")
  gui(p)

  return
end

end
