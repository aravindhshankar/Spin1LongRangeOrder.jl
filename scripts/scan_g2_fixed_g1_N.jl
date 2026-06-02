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
const VARIANCE_THRESHOLD = 1e-4
const G2_VALUES = collect(-0.3:0.05:0.3)
const N_VALUES = [16, 32, 64, 128, 256]

const LINKDIM_START_THRESH = [0, 50, 200, 400]
const LINKDIM_UPPER_CAP = [100, 200, 500, 700]
const LINKDIM_SCHEDULE_STEPS = [4, 8, 12, 12]

fmt(x) = replace(@sprintf(" % .3f", x), " " => "")

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
  noise = [1E-4, 1E-6, 0]
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
  figsavepath = joinpath("pngfigs", "N$(N)", "magz_vs_g2")
  mkpath(figsavepath)
  magfile = joinpath(figsavepath, "g1_$(fmt(g1))_magz_vs_g2.png")

  p = plot(g2_vals, magzs, marker=:circle, lw=2,
    xlabel="g2", ylabel="<Sz>",
    title="<Sz> per spin vs g2 for g1=$(g1)",
    legend=false)
  plot!(p, xgrid=true, ygrid=true)
  savefig(p, magfile)
  println("saved magz plot to $magfile")
end

function run_g2_scan_for_N(N, J, g1)
  sites = siteinds("S=1", N; conserve_sz=false)
  println("Running g2 scan for N=$N, g1=$g1")

  psi_prev = nothing
  magzs = Float64[]

  for (i, g2) in enumerate(G2_VALUES)
    println("\n==== g2 step $i / $(length(G2_VALUES)): g2=$g2 ====")

    if psi_prev === nothing
      psi_init = build_boundary_pinned_state(sites, J, g1, g2)
    else
      psi_init = psi_prev
    end

    energy, psi, var = run_g2_dmrg(sites, J, g1, g2, psi_init)
    println("Finished DMRG for g2=$g2, variance=$var")

    if var > VARIANCE_THRESHOLD
      println("Variance above threshold ($(VARIANCE_THRESHOLD)): retrying from pinned boundary state")
      psi_init = build_boundary_pinned_state(sites, J, g1, g2)
      energy, psi, var = run_g2_dmrg(sites, J, g1, g2, psi_init)
      println("Retry finished, variance=$var")
      if var > VARIANCE_THRESHOLD
        println("Warning: variance still above threshold after retry for g2=$g2")
      end
    end

    save_scan_state(psi, N, g1, g2)

    szvals = expect(psi, "Sz")
    magz = sum(szvals) / N
    push!(magzs, magz)
    psi_prev = psi
  end

  plot_magnetization(N, g1, G2_VALUES, magzs)
end
##
let
  idx = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
  N = N_VALUES[idx]
  J = -1.0
  g1 = G1_FIXED

  println("SLURM_ARRAY_TASK_ID=$idx -> N=$N, g1=$g1")
  run_g2_scan_for_N(N, J, g1)
end
