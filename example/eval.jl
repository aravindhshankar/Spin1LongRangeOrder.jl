using MKL
using Spin1LongRangeOrder
using ITensors, ITensorMPS
using Spin1LongRangeOrder.Hamiltonians
using HDF5
using Plots
using LinearAlgebra
gr()
##
let
  println(BLAS.get_config())
  N = 200
  sites = siteinds("S=1", N; conserve_sz=false)
  J = -1.0
  g = -0.5 #g>0 orders in X-Y plane, g=0 is the critical point, g<0 orders in Z direction
  g2 = -0.35 #g2 >0 favors X-Y plans, g2<0 favors Z axis
  #we saw g2 = -0.3 is in XY FM state
  #       g2 = -0.4 is in Z FM state (gapped)

  #   H_fm = build_fm_hamiltonian(sites, J)
  #   H_prelim = build_prelim_hamiltonian(sites, J)
  # H_pert = build_pert_hamiltonian(sites, J, g)
  H_pert = build_g1g2_hamiltonian(sites, J, g, g2)

  #   psi0 = random_mps(sites; linkdims=2)
  instate = ["X+" for n in 1:N]
  # instate = ["Up" for n in 1:N]
  psi_0 = random_mps(sites, instate; linkdims=10)
  # psi_fm = random_mps(sites, instate; linkdims=10)


  ### Load the recently stopped state and continue with the perturbed Hamiltonian
  #   psi_load = load_simulation("sampleMPS.h5")
  #   psi_load = replace_siteinds(psi_load, sites)  # Replace siteinds to match the new sites


  nsweeps = 6
  #   maxdim = [100, 150, 200, 200 ,200 ,200,200, 200, 200, 200, 220, 220, 220, 250, 250, 250, 250, 250, 300, 300, 300, 350, 350, 350, 400, 400] # gradually increase states kept
  maxdim = [50, 100, 200, 220]
  mindim = [10, 10, 2]
  eigsolve_krylovdim = 5
  cutoff = [1E-12]
  noise = [0, 1E-6, 0]
  obsparams = (energy_tol=1E-3, minsweeps=5, energy_type=Float64)
  observer = DMRGObserver(; obsparams...)
  # @profview dmrg(H_pert, psi_fm; nsweeps, maxdim, cutoff, mindim=mindim,
  #     eigsolve_krylovdim=eigsolve_krylovdim)
  energy, psi = dmrg(H_pert, psi_0; nsweeps, maxdim, cutoff, mindim,
    eigsolve_krylovdim, noise, observer)

  # writeToFile("mps_test")
  # save_simulation("sampleMPS.h5", psi)


  println("Energy density = ", energy/N)
  H2 = inner(H_pert, psi, H_pert, psi)
  E = inner(psi', H_pert, psi)
  var = H2 - E^2
  @show var
  zzcorr = correlation_matrix(psi, "Sz", "Sz")
  xxcorr = correlation_matrix(psi, "Sx", "Sx")
  yycorr = correlation_matrix(psi, "Sy", "Sy")
  pmcorr = correlation_matrix(psi, "S+", "S-")

  xplotvals = range(start=1, length=N, step=1)
  line1 = abs.(zzcorr[Int64(N // 2), :])
  line2 = abs.(xxcorr[Int64(N // 2), :])
  line3 = abs.(yycorr[Int64(N // 2), :])
  line4 = abs.(pmcorr[Int64(N // 2), :])
  p = plot(xplotvals, [line1, line2, line3, line4], ms=5, lw=2,
    label=["Sz-Sz" "Sx-Sx" "Sy-Sy" "+-"], xlabel="Distance", ylabel="Correlation",
    title="g1 = $(g) g2 = $(g2)", legend=:topright)
  plot!(xscale=:identity, yscale=:log10, minorgrid=true)
  # plot!(xscale=:identity, yscale=:identity, minorgrid=true)
  # plot!(xscale=:log10, yscale=:log10, minorgrid=true)
  #
  # savefig(p, "figure.png")
  # println("saved")
  return
end
