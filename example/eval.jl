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
  N = 100
  sites = siteinds("S=1", N; conserve_sz=false)
  h = 0.0
  J = -1.0
  g = 0.5

  H_fm = build_fm_hamiltonian(sites, J)
  H_prelim = build_prelim_hamiltonian(sites, J)
  H_pert = build_pert_hamiltonian(sites, J, g)

  nsweeps_p = 3
  maxdim_p = [5, 20, 50] # gradually increase states kept
  # mindim = [1, 1, 5, 10, 20, 50]
  mindim_p = [1]
  eigsolve_krylovdim_p = 3 #default 3
  cutoff_p = [1E-12] # desired truncation error
  noise_p = [1e-6]

  psi0 = random_mps(sites; linkdims=2)
  @show typeof(psi0)
  # instate = [isodd(n) ? "Up" : "Up" for n in 1:N]
  # psi0 = random_mps(sites, instate; linkdims=10)
  # psi_fm = random_mps(sites, instate; linkdims=10)
  energy, psi_p = dmrg(H_prelim, psi0; nsweeps=nsweeps_p, maxdim=maxdim_p,
    cutoff=cutoff_p, mindim=mindim_p,
    eigsolve_krylovdim=eigsolve_krylovdim_p, noise=noise_p)

  nsweeps = 2
  maxdim = [100]
  mindim = [1, 1]
  eigsolve_krylovdim = 3
  cutoff = [1E-12]
  noise = [0]
  energy, psi_fm = dmrg(H_fm, psi_p; nsweeps, maxdim, cutoff, mindim=mindim,
    eigsolve_krylovdim=eigsolve_krylovdim)


  ### Load the recently stopped state and continue with the perturbed Hamiltonian -- DOESN"T WORK
  # psi_fm_file= h5open("sampleMPS.h5", "r") do f
  #   read(f, "psi")
  # end
  # psi_fm = h5open("sampleMPS.h5", "r") do f
  #   read(f, "psi", ITensorMPS.MPS)
  # end
  @show typeof(psi_fm)
  
  nsweeps = 50
  maxdim = [100, 150, 200, 200 ,200 ,200,200, 200, 200, 200, 220]
  # maxdim = [200, 220, 250]
  mindim = [10, 10]
  eigsolve_krylovdim = 3
  cutoff = [1E-12]
  noise = [0]
# @profview dmrg(H_pert, psi_fm; nsweeps, maxdim, cutoff, mindim=mindim,
#     eigsolve_krylovdim=eigsolve_krylovdim)
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

  xplotvals = range(start=1, length=N, step=1)
  line1 = abs.(zzcorr[Int64(N // 2), :])
  line2 = abs.(xxcorr[Int64(N // 2), :])
  p = plot(xplotvals, [line1, line2], ms=5, lw=2, label=["Sz-Sz" "Sx-Sx"], xlabel="Distance", ylabel="Correlation",
     title="g = $(g)", legend=:topright)
  plot!(xscale=:identity, yscale=:log10, minorgrid=true)
  # plot!(xscale=:identity, yscale=:identity, minorgrid=true)
  # plot!(xscale=:log10, yscale=:log10, minorgrid=true)
  savefig(p, "figure.png")
  println("saved")
  gui(p)
  return 
end