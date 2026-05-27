using MKL
using Spin1LongRangeOrder
using ITensors, ITensorMPS
using Spin1LongRangeOrder.Hamiltonians
using HDF5
using Plots
using LinearAlgebra
using Printf
gr()
##
let
  println(BLAS.get_config())
  N = 20
  sites = siteinds("S=1", N; conserve_sz=false)
  J = -1.0
  g1 = -0.2 #g>0 orders in X-Y plane, g=0 is the critical point, g<0 orders in Z direction
  g2 = 0.25 #g2 >0 favors X-Y plans, g2<0 favors Z axis
  hx = 0.0
  hz = 0.0
  boundary_op = "Sz" # "Sx" or "Sz"
  boundary_h = -1.0
  #we saw g2 = -0.3 is in XY FM state
  #       g2 = -0.4 is in Z FM state (gapped)

  #   H_fm = build_fm_hamiltonian(sites, J)
  #   H_prelim = build_prelim_hamiltonian(sites, J)
  # H_pert = build_pert_hamiltonian(sites, J, g1)
  # H_pert = build_g1g2_hamiltonian(sites; J, g1, g2, hx, hz)
  H_pert = build_g1g2_hamiltonian(sites, J, g1, g2)

  #   psi0 = random_mps(sites; linkdims=2)
  instate = ["X+" for n in 1:N]
  # instate = ["Up" for n in 1:N]
  psi_0 = random_mps(sites, instate; linkdims=10)
  # psi_fm = random_mps(sites, instate; linkdims=10)


  ### Load the recently stopped state and continue with the perturbed Hamiltonian
  #   psi_load = load_simulation("sampleMPS.h5")
  #   psi_load = replace_siteinds(psi_load, sites)  # Replace siteinds to match the new sites


  ### Phase 1 : add a boundary annealing 
  n_anneal = 1
  for i in 1:n_anneal
    boundary_h_i = boundary_h * (1 - (i-1)/n_anneal)
    H_anneal = build_g1g2_hamiltonian(sites; J, g1, g2, boundary_op, boundary_h=boundary_h_i)
    energy, psi_0 = dmrg(H_anneal, psi_0; nsweeps=3, maxdim=[20], cutoff=1E-10)
    println("Anneal step $i: boundary h = $boundary_h_i, energy density = ", energy/N)
  end


  nsweeps = 20
  #   maxdim = [100, 150, 200, 200 ,200 ,200,200, 200, 200, 200, 220, 220, 220, 250, 250, 250, 250, 250, 300, 300, 300, 350, 350, 350, 400, 400] # gradually increase states kept
  maxdim = [20, 50, 50, 100, 150, 200, 220, 250, 300]
  mindim = [10, 10, 2]
  eigsolve_krylovdim = 5
  cutoff = [1E-12]
  noise = [0, 1E-6, 0]
  obsparams = (energy_tol=1E-5, minsweeps=3, energy_type=Float64)
  observer = DMRGObserver(; obsparams...)
  # @profview dmrg(H_pert, psi_fm; nsweeps, maxdim, cutoff, mindim=mindim,
  #     eigsolve_krylovdim=eigsolve_krylovdim)
  energy, psi = dmrg(H_pert, psi_0; nsweeps, maxdim, cutoff, mindim,
    eigsolve_krylovdim, noise, observer)

  # writeToFile("mps_test")
  save_simulation("sampleMPS.h5", psi)


  println("Energy density = ", energy/N)
  H2 = inner(H_pert, psi, H_pert, psi)
  E = inner(psi', H_pert, psi)
  var = H2 - E^2
  @show var
  zzcorr = correlation_matrix(psi, "Sz", "Sz")
  xxcorr = correlation_matrix(psi, "Sx", "Sx")
  yycorr = correlation_matrix(psi, "Sy", "Sy")
  pmcorr = correlation_matrix(psi, "S+", "S-")
  szvals = expect(psi, "Sz")
  magz = sum(szvals) / N
  magx = sum(expect(psi, "Sx")) / N
  magy = sum(expect(psi, "Sy")) / N
  println("---- Magnetization per site ----")
  @show magz
  @show magx
  @show magy
  println("--------------------------------")

  xplotvals = range(start=1, length=N, step=1)
  # startplot = Int64(N // 2)
  startplot = Int64(2)
  line1 = abs.(zzcorr[startplot, :])
  line2 = abs.(xxcorr[startplot, :])
  line3 = abs.(yycorr[startplot, :])
  line4 = abs.(pmcorr[startplot, :])
  p = plot(xplotvals, [line1, line2, line3, line4], ms=5, lw=2,
    label=["Sz-Sz" "Sx-Sx" "Sy-Sy" "+-"], xlabel="Distance", ylabel="Correlation",
      title="g1 = $(g1) g2 = $(g2) <Sz>=$(@sprintf("%.3f", magz)), <Sx>=$(@sprintf("%.3f", magx))", legend=:topright)
    plot!(xscale=:identity, yscale=:log10, minorgrid=true)
    # plot!(xscale=:identity, yscale=:identity, minorgrid=true)
  # plot!(xscale=:log10, yscale=:log10, minorgrid=true)
  #
  savefig(p, "figure.png")
  display(p)
  # println("saved")
  return
end #let
