using ITensors, ITensorMPS
using Plots
gr()

let
  N = 50
  sites = siteinds("S=1/2", N)
  h = 0.0
  J = -1.0 # - for FM, + for AFM

  jop = "Sx"
  hop = "Sz"
  os = OpSum()
  for j = 1:N-1
    os += 4 * J, jop, j, jop, j + 1
    os += 4 * J, "Sy", j, "Sy", j + 1 #add a YY term
    os += 2 * h, hop, j
  end
  os += 2 * h, hop, N
  # os += J,"Sz",N,"Sz",1 #PBC
  H = MPO(os, sites)

  nsweeps = 10 # number of sweeps is 5
  maxdim = [5, 10, 20, 50, 100, 200] # gradually increase states kept
  mindim = [2, ]
  eigsolve_krylovdim = 10 #default 3
  cutoff = [1E-12] # desired truncation error

  psi0 = random_mps(sites; linkdims=10)

  energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, mindim=mindim,
    eigsolve_krylovdim=eigsolve_krylovdim)
  println("Energy = ", energy)

  H2 = inner(H, psi, H, psi)   # Computes <H^2>
  E = inner(psi', H, psi)      # Computes <H>
  var = H2 - E^2               # Computes Variance
  @show var 

  zzcorr = correlation_matrix(psi, "Sz", "Sz")
  magz = 2 * sum(expect(psi, "Sz")) / N 
  magx = 2 * sum(expect(psi, "Sx")) / N 
  @show magz 
  @show magx 
  # maxy = maximum(zzcorr)
  # miny = minimum(zzcorr)
  # @show(zzcorr[1,:])
  p = plot(abs.(zzcorr[Int64(N // 2), :]), ms=5, lw=2)
  # plot!(xscale=:identity, yscale=:log10, minorgrid=true)
  plot!(xscale=:log10, yscale=:log10, minorgrid=true)
  gui(p)

  return
end
