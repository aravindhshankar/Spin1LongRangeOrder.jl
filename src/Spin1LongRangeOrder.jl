module Spin1LongRangeOrder

using ITensors, ITensorMPS
using Plots
gr()

let
  N = 10
  sites = siteinds("S=1", N)
  h = 0.0
  J = -1.0

  os = OpSum()
  for j = 1:N-1
    os += J, "Sz", j, "Sz", j + 1
    os += J, "Sx", j, "Sx", j + 1
    os += J, "Sy", j, "Sy", j + 1
    os += h, "Sx", j
  end
  os += 2 * h, "Sx", N
  # os += J,"Sz",N,"Sz",1 #PBC
  H = MPO(os, sites)

  nsweeps = 7
  maxdim = [5, 10, 20, 50, 100, 200] # gradually increase states kept
  mindim = [1, 1, 5, 10, 20, 50, 100]
  eigsolve_krylovdim = 3 #default 3
  cutoff = [1E-12] # desired truncation error

  psi0 = random_mps(sites; linkdims=2)

  energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, mindim=mindim,
    eigsolve_krylovdim=eigsolve_krylovdim)
  println("Energy = ", energy)
  zzcorr = correlation_matrix(psi, "Sz", "Sz")

  p = plot(abs.(zzcorr[Int64(N // 2), :]), ms=5, lw=2)
  # plot!(xscale=:identity, yscale=:log10, minorgrid=true)
  plot!(xscale=:log10, yscale=:log10, minorgrid=true)
  gui(p)

  return
end

end
