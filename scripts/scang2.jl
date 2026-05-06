using MKL
using Spin1LongRangeOrder
using Spin1LongRangeOrder.Hamiltonians
using ITensorMPS
using Printf
using Plots

gr()


function runDMRGaction(hamparams)
  N = 100
  # hamparams = (J=1, g1=0.5, g2=-0.5)
  sites = siteinds("S=1", N)
  Ham = build_g1g2_hamiltonian(sites; hamparams...)
  println("Built hamiltonian")
  @show hamparams

  # psi_0 = randomMPS(sites, linkdims=10)
  # instate = ["Up" for n in 1:N]
  instate = ["X+" for _ in 1:N]
  psi_0 = random_mps(sites, instate; linkdims=10)

  nsweeps = 20
  maxdim = [20, 50, 100, 100, 150, 200, 200, 220, 250, 300]
  mindim = [10, 10, 10, 1]
  eigsolve_krylovdim = 5
  cutoff = [1E-12]
  noise = [1E-6, 0]
  obsparams = (energy_tol=1E-4, minsweeps=5, energy_type=Float64)
  observer = DMRGObserver(; obsparams...)

  energy, psi = dmrg(Ham, psi_0; nsweeps, maxdim, cutoff, mindim=mindim,
    eigsolve_krylovdim=eigsolve_krylovdim, observer, noise)

  fmt(x) = @sprintf("%.3f", x)
  savepath = "data/N$(N)/"
  mkpath(savepath)
  parts = [String(k) * "_" * fmt(v) for (k, v) in pairs(hamparams)]
  filename = savepath * join(parts, "") * ".h5"
  save_simulation(filename, psi, Dict(pairs(hamparams)))
  println("saved to filename : " * filename)

  #-----------------------------------------savefig--------------
  figsavepath = "pngfigs/N$(N)/"
  mkpath(figsavepath)
  figfilename = figsavepath * join(parts, "") * ".png"
  H2 = inner(Ham, psi, Ham, psi)
  E = inner(psi', Ham, psi)
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
    title="g1 = $(hamparams.g1) g2 = $(hamparams.g2)", legend=:topright)
  plot!(xscale=:identity, yscale=:log10, minorgrid=true)
  savefig(p, figfilename)
  println("saved figure to file: " * figfilename)

  return
end

let
  gval = 0.5
  g2list = [(J=-1, g1=gval, g2=g2val)
            for g2val in range(start=-0.4, step=0.01, length=10)]
  #stops at -0.31
  # idx = 1
  idx = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
  hamparams = g2list[idx]
  runDMRGaction(hamparams)
end


