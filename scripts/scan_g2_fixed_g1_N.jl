using MKL
using Spin1LongRangeOrder
using Spin1LongRangeOrder.Hamiltonians
using ITensors, ITensorMPS
using HDF5
using Plots
using LinearAlgebra
using Printf
##
gr()

const G1_FIXED = -0.2
const VARIANCE_THRESHOLD = 1e-3
const G2_VALUES = collect(0.1:0.01:0.35)
const N_VALUES = [16, 32, 64, 100, 128, 256]
const RESTART_G2_START = Dict(64 => 0.20, 100 => 0.18, 128 => 0.16, 256 => 0.15)  # e.g., Dict(16 => 0.25, 32 => 0.20) to restart from specific g2

const LINKDIM_START_THRESH = [0, 100, 400]
const LINKDIM_UPPER_CAP = [220, 500, 800]
const LINKDIM_SCHEDULE_STEPS = [16, 12, 12]

fmt(x) = replace(@sprintf(" % .3f", x), " " => "")

function fig_dir(N, subdir="")
  base = joinpath("pngfigs", "N$(N)")
  path = subdir == "" ? base : joinpath(base, subdir)
  mkpath(path)
  return path
end

function build_boundary_pinned_state(sites, J, g1, g2;
    boundary_op="Sx", boundary_h=-10.0, n_anneal=1)
  instate = ["X+" for _ in 1:length(sites)]
  psi = random_mps(sites, instate; linkdims=10)

  for i in 1:n_anneal
    boundary_h_i = boundary_h * (1 - (i - 1) / n_anneal)
    Hpin = build_g1g2_hamiltonian(sites, J, g1, g2,
      boundary_op, boundary_h_i)
    energy, psi = dmrg(Hpin, psi; nsweeps=3, maxdim=[20], cutoff=1e-10)
    println("Anneal step $i: boundary h = $boundary_h_i, energy density = ", energy / length(sites))
  end

  return psi
end

function build_dmrg_observer(etol=1E-6)
   minsweeps = 7
   obsparams = (energy_tol=etol, minsweeps=minsweeps, energy_type=Float64)
   return DMRGObserver(; obsparams...)
end

function try_load_prev_g2_state(N, g1, g2)
  idx = findfirst(x -> x ≈ g2, G2_VALUES)
  if idx === nothing || idx == 1
    return nothing
  end
  
  prev_g2 = G2_VALUES[idx - 1]
  filename = joinpath("data", "N$(N)", "g1_$(fmt(g1))_g2_$(fmt(prev_g2)).h5")
  
  if isfile(filename)
    psi = load_simulation(filename)
    println("Loaded initial state from previous g2=$prev_g2")
    return psi
  end
  
  return nothing
end



function build_maxdim_schedule(max_bd)
  idx = findlast(x -> max_bd >= x, LINKDIM_START_THRESH)
  idx === nothing && (idx = 1)

  cap = LINKDIM_UPPER_CAP[idx]
  steps = LINKDIM_SCHEDULE_STEPS[idx]

  if max_bd >= cap
    return [max_bd]
  end

  return unique(Int.(round.(range(max_bd, cap; length=steps))))
end

function run_g2_dmrg(sites, J, g1, g2, psi_init)
  H = build_g1g2_hamiltonian(sites, J, g1, g2)

  nsweeps = 70
  max_bd = maximum([linkdim(psi_init, i) for i in 1:length(psi_init)-1])
  maxdim = build_maxdim_schedule(max_bd)
  mindim = [2]
  eigsolve_krylovdim = 5
  cutoff = [1E-12]
  noise = [1E-3, 1E-4, 1E-5, 1E-6, 0]
  observer = build_dmrg_observer()

  energy, psi = dmrg(H, psi_init; nsweeps, maxdim, cutoff, mindim,
    eigsolve_krylovdim, noise, observer)

  H2 = inner(H, psi, H, psi)
  E = inner(psi', H, psi)
  var = H2 - E^2

  return energy, psi, var
end

function save_scan_state(psi, N, g1, g2)
  savepath = joinpath("data", "N$(N)")
  mkpath(savepath)
  filename = joinpath(savepath, "g1_$(fmt(g1))_g2_$(fmt(g2)).h5")
  save_simulation(filename, psi, Dict("J" => -1.0, "g1" => g1, "g2" => g2))
  println("saved state to $filename")
end

function plot_magnetization(N, g1, g2_vals, magzs)
  figsavepath = fig_dir(N, "magz_vs_g2")
  magfile = joinpath(figsavepath, "g1_$(fmt(g1))_magz_vs_g2.png")

  p = plot(g2_vals, magzs, marker=:circle, lw=2,
    xlabel="g2", ylabel="<Sz>",
    title="<Sz> per spin vs g2 for g1=$(g1)",
    legend=false)
  plot!(p, xgrid=true, ygrid=true)
  savefig(p, magfile)
  println("saved magz plot to $magfile")
end

function run_g2_scan_for_N(N, J, g1; start_g2=G2_VALUES[1])
  sites = siteinds("S=1", N; conserve_sz=false)
  println("Running g2 scan for N=$N, g1=$g1, starting from g2=$start_g2")

  psi_prev = nothing
  magzs = Float64[]

  start_idx = findfirst(x -> x ≈ start_g2, G2_VALUES)
  start_idx === nothing && error("start_g2=$start_g2 not found in G2_VALUES")

  for (i, g2) in enumerate(G2_VALUES[start_idx:end])
    global_idx = start_idx + i - 1
    println("\n==== g2 step $global_idx / $(length(G2_VALUES)): g2=$g2 ====")

    if psi_prev === nothing
      psi_init = try_load_prev_g2_state(N, g1, g2)
      if psi_init === nothing
        psi_init = build_boundary_pinned_state(sites, J, g1, g2)
      else
        psi_init = ITensorMPS.replace_siteinds(psi_init, sites)  # Align site indices if loaded from previous g2
        # psi_init = Spin1LongRangeOrder.replace_siteinds(psi_init, sites) # Align site indices if loaded from previous g2
      end
    else
      psi_init = psi_prev
    end

    energy, psi, var = run_g2_dmrg(sites, J, g1, g2, psi_init)
    println("Finished DMRG for g2=$g2, variance=$var, energy density = ", energy / N)

    if var > VARIANCE_THRESHOLD
      println("Variance above threshold ($(VARIANCE_THRESHOLD)): retrying from pinned boundary state")
      psi_init = build_boundary_pinned_state(sites, J, g1, g2)
      energy, psi, var = run_g2_dmrg(sites, J, g1, g2, psi_init)
      println("Retry finished, variance=$var, energy density = ", energy / N)
      if var > VARIANCE_THRESHOLD
        println("Warning: variance still above threshold after retry for g2=$g2")
      end
    end

    save_scan_state(psi, N, g1, g2)

    # Correlation matrices and magnetizations
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

    # Save correlation plot
    xplotvals = range(start=1, length=N, step=1)
    startplot = Int(2)
    line1 = abs.(zzcorr[startplot, :])
    line2 = abs.(xxcorr[startplot, :])
    line3 = abs.(yycorr[startplot, :])
    line4 = abs.(pmcorr[startplot, :])

    figsavepath = fig_dir(N)
    figfilename = joinpath(figsavepath, "g1_$(fmt(g1))_g2_$(fmt(g2))_corr.png")

    p = plot(xplotvals, [line1, line2, line3, line4], ms=5, lw=2,
      label=["Sz-Sz" "Sx-Sx" "Sy-Sy" "+-"],
      xlabel="Distance", ylabel="Correlation",
      title="g1 = $(g1) g2 = $(g2) <Sz>=$(round(magz, digits=3)), <Sx>=$(round(magx, digits=3))",
      legend=:topright)
    plot!(xscale=:identity, yscale=:log10, minorgrid=true)

    savefig(p, figfilename)
    println("saved figure to $figfilename")

    push!(magzs, magz)
    psi_prev = psi
  end

  plot_magnetization(N, g1, G2_VALUES, magzs)
end
##
let
  idx = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
  # idx = 3
  N = N_VALUES[idx]
  @show N
  J = -1.0
  g1 = G1_FIXED
  start_g2 = get(RESTART_G2_START, N, G2_VALUES[1])

  println("SLURM_ARRAY_TASK_ID=$idx -> N=$N, g1=$g1, start_g2=$start_g2")
  run_g2_scan_for_N(N, J, g1; start_g2=start_g2)
end
