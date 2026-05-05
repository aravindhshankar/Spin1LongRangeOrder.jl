using MKL
using Spin1LongRangeOrder
using ITensors, ITensorMPS
using Spin1LongRangeOrder.Hamiltonians
using HDF5
using Plots
gr()
##
let
  # println(BLAS.get_config())  
  N = 200
  sites = siteinds("S=1", N; conserve_sz=false)
  h = 0.0
  J = -1.0
  g = 1.5 #g>0 orders in X-Y plane, g=0 is the critical point, g<0 orders in Z direction

#   H_fm = build_fm_hamiltonian(sites, J)
#   H_prelim = build_prelim_hamiltonian(sites, J)
  H_pert = build_pert_hamiltonian(sites, J, g)

#   nsweeps_p = 3
#   maxdim_p = [5, 20, 50] # gradually increase states kept
#   # mindim = [1, 1, 5, 10, 20, 50]
#   mindim_p = [1]
#   eigsolve_krylovdim_p = 3 #default 3
#   cutoff_p = [1E-12] # desired truncation error
#   noise_p = [1e-6]

#   psi0 = random_mps(sites; linkdims=2)
    instate = ["X+" for n in 1:N]
    # instate = ["Up" for n in 1:N]
    psi_0 = random_mps(sites, instate; linkdims=10)
    # psi_fm = random_mps(sites, instate; linkdims=10)

#   energy, psi_p = dmrg(H_prelim, psi_0; nsweeps=nsweeps_p, maxdim=maxdim_p,
#     cutoff=cutoff_p, mindim=mindim_p,
#     eigsolve_krylovdim=eigsolve_krylovdim_p, noise=noise_p)

#   nsweeps = 2
#   maxdim = [100]
#   mindim = [1, 1]
#   eigsolve_krylovdim = 3
#   cutoff = [1E-12]
#   noise = [0]
#   energy, psi_fm = dmrg(H_fm, psi_p; nsweeps, maxdim, cutoff, mindim=mindim,
#     eigsolve_krylovdim=eigsolve_krylovdim)


  ### Load the recently stopped state and continue with the perturbed Hamiltonian
#   psi_load = load_simulation("sampleMPS.h5")
#   psi_load = replace_siteinds(psi_load, sites)  # Replace siteinds to match the new sites

  
  nsweeps = 50
#   maxdim = [100, 150, 200, 200 ,200 ,200,200, 200, 200, 200, 220, 220, 220, 250, 250, 250, 250, 250, 300, 300, 300, 350, 350, 350, 400, 400] # gradually increase states kept
  maxdim = [100, 200, 220, 250, 300, 350]
  mindim = [10, 10]
  eigsolve_krylovdim = 5
  cutoff = [1E-12]
  noise = [0]
# @profview dmrg(H_pert, psi_fm; nsweeps, maxdim, cutoff, mindim=mindim,
#     eigsolve_krylovdim=eigsolve_krylovdim)
  energy, psi = dmrg(H_pert, psi_0; nsweeps, maxdim, cutoff, mindim=mindim,
    eigsolve_krylovdim=eigsolve_krylovdim)

  # writeToFile("mps_test")
  save_simulation("sampleMPS.h5", psi)
  

  println("Energy = ", energy)
  H2 = inner(H_pert, psi, H_pert, psi)
  E = inner(psi', H_pert, psi)
  var = H2 - E^2
  @show var
  zzcorr = correlation_matrix(psi, "Sz", "Sz")
  xxcorr = correlation_matrix(psi, "Sx", "Sx")
  yycorr = correlation_matrix(psi, "Sy", "Sy")

  xplotvals = range(start=1, length=N, step=1)
  line1 = abs.(zzcorr[Int64(N // 2), :])
  line2 = abs.(xxcorr[Int64(N // 2), :])
  line3 = abs.(yycorr[Int64(N // 2), :])
  p = plot(xplotvals, [line1, line2, line3], ms=5, lw=2, label=["Sz-Sz" "Sx-Sx" "Sy-Sy"], xlabel="Distance", ylabel="Correlation",
     title="g = $(g)", legend=:topright)
  plot!(xscale=:identity, yscale=:log10, minorgrid=true)
  # plot!(xscale=:identity, yscale=:identity, minorgrid=true)
  # plot!(xscale=:log10, yscale=:log10, minorgrid=true)
  savefig(p, "figure.png")
  println("saved")
  gui(p)
  return 
end