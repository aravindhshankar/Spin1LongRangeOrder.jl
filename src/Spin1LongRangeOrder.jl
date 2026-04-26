module Spin1LongRangeOrder

using ITensors, ITensorMPS
using Plots
using HDF5
gr()

let
  N = 100
  sites = siteinds("S=1", N; conserve_sz=false)
  h = 0.0
  J = -1.0
  g = 1.5

  os = OpSum()
  for j = 1:N-1
    os += J, "Sz", j, "Sz", j + 1
    os += 0.5 * J, "S+", j, "S-", j + 1
    os += 0.5 * J, "S-", j, "S+", j + 1
    # os += 0.01, "Sz", j
  end
  H_fm = MPO(os, sites)

  os_prelim = OpSum()
  for j = 1:N-1
    os_prelim += J, "Sz", j, "Sz", j + 1
    os_prelim += 0.5 * J, "S+", j, "S-", j + 1
    os_prelim += 0.5 * J, "S-", j, "S+", j + 1
    os_prelim += 1, "Sz", j
  end
  H_prelim = MPO(os_prelim, sites)

  os_pert = OpSum()
  for j = 1:N-1
    os_pert += J, "Sz", j, "Sz", j + 1
    os_pert += 0.5 * J, "S+", j, "S-", j + 1
    os_pert += 0.5 * J, "S-", j, "S+", j + 1
    os_pert += g, "Sz2", j
  end
  H_pert = MPO(os_pert, sites)

  nsweeps_p = 3
  maxdim_p = [5, 20, 50] # gradually increase states kept
  # mindim = [1, 1, 5, 10, 20, 50]
  mindim_p = [1]
  eigsolve_krylovdim_p = 3 #default 3
  cutoff_p = [1E-12] # desired truncation error
  noise_p = [1e-6]

  psi0 = random_mps(sites; linkdims=2)
  # instate = [isodd(n) ? "Up" : "Up" for n in 1:N]
  # psi0 = random_mps(sites, instate; linkdims=10)
  # psi_fm = random_mps(sites, instate; linkdims=10)
  energy, psi_p = dmrg(H_prelim, psi0; nsweeps=nsweeps_p, maxdim=maxdim_p,
    cutoff=cutoff_p, mindim=mindim_p,
    eigsolve_krylovdim=eigsolve_krylovdim_p, noise=noise_p)

  psi_fm = psi0

  nsweeps = 2
  maxdim = [100]
  mindim = [1, 1]
  eigsolve_krylovdim = 3
  cutoff = [1E-12]
  noise = [0]
  energy, psi_fm = dmrg(H_fm, psi_p; nsweeps, maxdim, cutoff, mindim=mindim,
    eigsolve_krylovdim=eigsolve_krylovdim)

  nsweeps = 10
  maxdim = [20, 40, 60, 80, 100, 150, 200]
  mindim = [10, 10]
  eigsolve_krylovdim = 3
  cutoff = [1E-12]
  noise = [0]
  energy, psi = dmrg(H_pert, psi_fm; nsweeps, maxdim, cutoff, mindim=mindim,
    eigsolve_krylovdim=eigsolve_krylovdim)

  # writeToFile("mps_test")
  f = h5open("sampleMPS.h5", "w") do f
    write(f, "psi", psi)
  end

  println("Energy = ", energy)
  H2 = inner(H_pert, psi, H_pert, psi)
  E = inner(psi', H_pert, psi)
  var = H2 - E^2
  @show var
  zzcorr = correlation_matrix(psi, "Sz", "Sz")
  xxcorr = correlation_matrix(psi, "Sx", "Sx")

  xplotvals = range(start=1, length=100, step=1)
  line1 = abs.(zzcorr[Int64(N // 2), :])
  line2 = abs.(xxcorr[Int64(N // 2), :])
  p = plot(xplotvals, [line1, line2], ms=5, lw=2)
  plot!(xscale=:identity, yscale=:log10, minorgrid=true)
  # plot!(xscale=:identity, yscale=:identity, minorgrid=true)
  # plot!(xscale=:log10, yscale=:log10, minorgrid=true)
  savefig(p, "figure.png")
  println("saved")
  gui(p)

  return
end

end
