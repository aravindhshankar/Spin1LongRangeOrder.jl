using MKL
using Spin1LongRangeOrder
using Spin1LongRangeOrder.Hamiltonians
using ITensors, ITensorMPS
using HDF5
using Plots
using LinearAlgebra
using Printf

gr()

function run_scan(N, J, g1, g2)
  println("Running scan for N=$N, J=$J, g1=$g1, g2=$g2")  
  sites = siteinds("S=1", N; conserve_sz=false)
  boundary_op = "Sx"
  boundary_h = -10.0

  instate = ["X+" for _ in 1:N]
  psi = random_mps(sites, instate; linkdims=10)

  # boundary annealing
  n_anneal = 1
  for i in 1:n_anneal
    boundary_h_i = boundary_h * (1 - (i - 1) / n_anneal)
    H_anneal = build_g1g2_hamiltonian(sites; J=J, g1=g1, g2=g2,
      boundary_op=boundary_op, boundary_h=boundary_h_i)
    energy, psi = dmrg(H_anneal, psi; nsweeps=3, maxdim=[20], cutoff=1e-10)
    println("Anneal step $i: boundary h = $boundary_h_i, energy density = ", energy / N)
  end

  H_pert = build_g1g2_hamiltonian(sites, J, g1, g2)

  nsweeps = 30
  maxdim = [20, 50, 100, 100, 150, 200, 220, 250, 300, 350]
  mindim = [10, 10, 2]
  eigsolve_krylovdim = 5
  cutoff = [1E-12]
  noise = [0, 1E-6, 0]
  obsparams = (energy_tol=1E-6, minsweeps=3, energy_type=Float64)
  observer = DMRGObserver(; obsparams...)

  energy, psi = dmrg(H_pert, psi; nsweeps, maxdim, cutoff, mindim,
    eigsolve_krylovdim, noise, observer)

  fmt(x) = replace(@sprintf("% .3f", x), " " => "")
  savepath = "data/N$(N)/"
  mkpath(savepath)
  filename = joinpath(savepath, "g1_$(fmt(g1))_g2_$(fmt(g2)).h5")
  save_simulation(filename, psi, Dict("J" => J, "g1" => g1, "g2" => g2))
  println("saved state to $filename")

  H2 = inner(H_pert, psi, H_pert, psi)
  E = inner(psi', H_pert, psi)
  var = H2 - E^2

  println(" ------ Variance ----------")
  println("Energy density = ", energy / N)
  @show var
  println("---------------------------")

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
  startplot = Int64(2)
  line1 = abs.(zzcorr[startplot, :])
  line2 = abs.(xxcorr[startplot, :])
  line3 = abs.(yycorr[startplot, :])
  line4 = abs.(pmcorr[startplot, :])

  figsavepath = "pngfigs/N$(N)/"
  mkpath(figsavepath)
  figfilename = joinpath(figsavepath, "g1_$(fmt(g1))_g2_$(fmt(g2)).png")

  p = plot(xplotvals, [line1, line2, line3, line4], ms=5, lw=2,
    label=["Sz-Sz" "Sx-Sx" "Sy-Sy" "+-"],
    xlabel="Distance", ylabel="Correlation",
    title="g1 = $(g1) g2 = $(g2) <Sz>=$(round(magz, digits=3)), <Sx>=$(round(magx, digits=3))",
    legend=:topright)
  plot!(xscale=:identity, yscale=:log10, minorgrid=true)

  savefig(p, figfilename)
  println("saved figure to $figfilename")

  return magz
end

let
  N = 20
  J = -1.0
  g1vals = [-0.2, -0.1, 0.1, 0.2]
  g2vals = collect(-0.3:0.05:0.3)

  idx = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
  g1 = g1vals[idx]
  println("Running g1 = $g1 on SLURM_ARRAY_TASK_ID=$idx")

  magzs = Float64[]
  for g2 in g2vals
    println("Starting scan for g1=$g1, g2=$g2")
    push!(magzs, run_scan(N, J, g1, g2))
  end

  fmt(x) = replace(@sprintf("% .3f", x), " " => "")
  figsavepath = "pngfigs/N$(N)/magz_vs_g2/"
  mkpath(figsavepath)
  magfile = joinpath(figsavepath, "g1_$(fmt(g1))_magz_vs_g2.png")

  p = plot(g2vals, magzs, marker=:circle, lw=2,
    xlabel="g2", ylabel="<Sz>",
    title="<Sz> per spin vs g2 for g1=$(g1)",
    legend=false)
  plot!(p, xgrid=true, ygrid=true)

  savefig(p, magfile)
  println("saved magz plot to $magfile")
end
