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

# FM instability condition (paper, coefficient a of Eq. 7):
#   a ∝ Σ_j j^2 (V^{+-}_j - V^{++}_j)  > 0
#   i.e. Vpm > Vpp drives the ferromagnetic transition.
#-------------------------------------------------------------------------------


# ─── Sweep schedule ───────────────────────────────────────────────────────────
function make_sweeps(max_sweeps=50)
  # Each entry: (maxdim, cutoff, noise)
  # The schedule has 50 rows; bond dim and noise saturate after row 6.
  base = [
    ("maxdim", "cutoff", "noise"),
    fill((20, 1e-8, 1e-4), 2)...,
    fill((50, 1e-8, 1e-6), 4)...,
    fill((100, 1e-10, 1e-8), 5)...,
    fill((200, 1e-12, 0.0), 10)...,
    (400, 1e-10, 0.0),
    # (500,  1e-12, 0.0),
  ]
  sw = Sweeps(max_sweeps, stack(base, dims=1))
  return sw
end

function make_sweeps(max_sweeps=50, init_dim=10)
  base = [
    ("maxdim", "cutoff", "noise"),
    fill((20, 1e-12, 1e-6), 2)...,
    fill((50, 1e-8, 1e-6), 4)...,
    fill((100, 1e-10, 1e-8), 5)...,
    fill((200, 1e-12, 0.0), 10)...,
    (400, 1e-10, 0.0),
  ]
  # idx = findlast(x -> init_dim > x[1], base[2:end])
  # val = nothing
  # if !isnothing(idx)
  #   val = base[2:end][idx][1]
  #   deleteat!(base, 2:idx+1)
  # end
  deleteat!(base, 1 .+ findall(x -> init_dim >= x[1], base[2:end]))
  sw = Sweeps(max_sweeps, stack(base, dims=1))
  return sw
end

# ─── Early-stopping observer ──────────────────────────────────────────────────
# Halts DMRG when |ΔE| < energy_tol for one full sweep, after min_sweeps.
mutable struct EnergyObserver <: AbstractObserver
  energy_tol::Float64
  min_sweeps::Int
  last_energy::Float64
  sweep_count::Int
end

EnergyObserver(; energy_tol=1e-4, min_sweeps=3) =
  EnergyObserver(energy_tol, min_sweeps, Inf, 0)

function ITensorMPS.checkdone!(obs::EnergyObserver; energy, sweep, kwargs...)
  obs.sweep_count = sweep
  converged = abs(energy - obs.last_energy) < obs.energy_tol &&
              sweep >= obs.min_sweeps
  obs.last_energy = energy
  if converged
    println("  ── EnergyObserver: converged at sweep $sweep " *
            "(|ΔE| < $(obs.energy_tol))  ──")
  end
  return converged
end

# ─── Build MPO ────────────────────────────────────────────────────────────────
function build_hamiltonian(sites, t, U, Vpp, Vpm; bpin=0)
  N = length(sites)
  os = OpSum()

  for i in 1:(N-1)
    # NN hopping (both spins)
    os += -t, "Cdagup", i, "Cup", i + 1
    os += t, "Cup", i, "Cdagup", i + 1   # h.c.
    os += -t, "Cdagdn", i, "Cdn", i + 1
    os += t, "Cdn", i, "Cdagdn", i + 1   # h.c.

    # V^{++}: same-spin NN repulsion (↑↑ and ↓↓)
    os += Vpp, "Nup", i, "Nup", i + 1
    os += Vpp, "Ndn", i, "Ndn", i + 1

    # V^{+-}: opposite-spin NN repulsion (↑↓ and ↓↑)
    os += Vpm, "Nup", i, "Ndn", i + 1
    os += Vpm, "Ndn", i, "Nup", i + 1

    # On-site Hubbard U
    os += U, "Nupdn", i

    if abs(bpin) > 1e-14
      os += bpin, "Cdagup", i, "Cdagdn", i
      os += bpin, "Cdn", i, "Cup", i
    end
  end

  os += U, "Nupdn", N  # On-site U for the last site
  # if abs(bpin) > 1e-14
  #     opvec = ["Cup", "Cdn", "Cdagup", "Cdagdn"]
  #     for opair in Iterators.product(opvec, opvec)
  #         os += bpin, opair[1], 1, opair[2], 1
  #         os += bpin, opair[1], N, opair[2], N 
  #     end
  # end

  return MPO(os, sites)
end

# ─── Initial MPS (alternating ↑↓, half filling) ──────────────────────────────
function initial_mps(sites; Npart=nothing)
  N = length(sites)
  # state = [isodd(i) ? "Up" : "Dn" for i in 1:N]
  # return MPS(sites, state)
  Npart = isnothing(Npart) ? length(sites) : Npart
  state = ["Emp" for n in 1:N]
  p = Npart
  for i in N:-1:1
    if p > i
      # println("Doubly occupying site $i")
      state[i] = "UpDn"
      p -= 2
    elseif p > 0
      # println("Singly occupying site $i")
      state[i] = (isodd(i) ? "Up" : "Dn")
      p -= 1
    end
  end
  # Initialize wavefunction to be bond
  # dimension 10 random MPS with number
  # of particles the same as `state`
  psi0 = random_mps(sites, state; linkdims=10)

  # Check total number of particles:
  # @show flux(Spsi0)
  return psi0
end

# ─── Run a single DMRG calculation ───────────────────────────────────────────
function run_dmrg(N, t, U, Vpp, Vpm; sites=nothing, initial_psi=nothing, sweepcount=100, bpin=0, verbose=true)
  sites = isnothing(sites) ? siteinds("Electron", N; conserve_nf=false, conserve_sz=false) : sites
  # sites = siteinds("Electron", N; conserve_qns=true)
  H = build_hamiltonian(sites, t, U, Vpp, Vpm; bpin)
  psi0 = isnothing(initial_psi) ? initial_mps(sites) : initial_psi
  sw = make_sweeps(sweepcount)

  obs = EnergyObserver(energy_tol=1e-8, min_sweeps=15)
  eigsolve_krylovdim = 10
  @time E, psi = dmrg(H, psi0, sw; eigsolve_krylovdim, outputlevel=verbose ? 1 : 0, observer=obs)
  var = variance_gs(H, psi)
  @printf("\nGround state energy variance:  = %.2e\n", var)
  print("Total electron density: ⟨N⟩ = ")
  number_profile = expect(psi, "Ntot")
  # @show number_profile
  total_N = sum(number_profile)
  @printf("%.4f  (expected %d)\n", total_N, N)
  return E, psi, sites
end

# ─── Observables ─────────────────────────────────────────────────────────────
function measure(psi, sites; label="")
  N = length(sites)
  Sz = expect(psi, "Sz")
  Nup_v = expect(psi, "Nup")
  Ndn_v = expect(psi, "Ndn")
  Ntot = expect(psi, "Ntot")
  double = expect(psi, "Nupdn")

  total_Sz = sum(Sz)
  Sz_per_spin = 0.5 * sum(Nup_v - Ndn_v ./ Ntot)
  @show Sz_per_spin
  total_N = sum(Ntot)
  mean_double = sum(double) / N
  total_Sz_per_spin = total_Sz / total_N

  label == "" || println("\n── $label ──")
  @printf("  Total ⟨Sz⟩     = %+.6f\n", total_Sz)
  @printf("  Total ⟨Sz⟩ per spin     = %+.6f\n", total_Sz_per_spin)
  @printf("  Total ⟨N⟩      = %.4f  (expected %d)\n", total_N, N)
  @printf("  Mean ⟨n↑n↓⟩    = %.6f  (double occupancy)\n", mean_double)

  # println("\n  Site-resolved observables:")
  # println("  ", "-"^55)
  # @printf("  %4s  %8s  %8s  %8s  %8s\n", "site", "⟨Sz⟩", "⟨n↑⟩", "⟨n↓⟩", "⟨ntot⟩")
  # println("  ", "-"^55)
  # for i in 1:N
  #     @printf("  %4d  %+8.5f  %8.5f  %8.5f  %8.5f\n",
  #             i, Sz[i], Nup_v[i], Ndn_v[i], Ntot[i])
  # end

  # Spin-spin correlations from the central site
  # mid  = N ÷ 2
  SzSz = correlation_matrix(psi, "Sz", "Sz")
  # println("\n  ⟨Sz_$(mid) Sz_j⟩ spin correlations:")
  # for j in 1:N
  #     @printf("    j=%2d: %+.6f\n", j, SzSz[mid, j])
  # end

  # Spin structure factor S(q) = (1/N) Σ_{ij} cos(q(i-j)) ⟨Sz_i Sz_j⟩
  # println("\n  Spin structure factor S(q):")
  # for q in range(0, π, length=9)
  # Sq = sum(cos(q*(i-j)) * SzSz[i,j] for i in 1:N, j in 1:N) / N
  # @printf("    q/π = %.3f:  S(q) = %+.6f\n", q/π, Sq)
  # end

  if abs(total_Sz) > 0.3 * N / 2
    println("\n  >>> Strong ferromagnetic order detected!")
  elseif maximum(abs.(vec(sum(SzSz, dims=2)))) > 2.0
    println("\n  >>> Large uniform spin correlations — near FM critical point.")
  else
    println("\n  >>> No ferromagnetic order at these parameters.")
  end

  return Sz, SzSz, total_N
end



# ─── Scan ΔV = V^{+-} - V^{++} to locate the FM transition ──────────────────
function scan_deltaV(N=16, t=1.0, U=4.0, Vpp=0.5;
  dV_range=0.0:0.25:2.0, Npart=nothing)
  println("\n", "="^65)
  println("Scanning ΔV = V^{+-} - V^{++} to locate FM transition")
  @printf("  N=%d  t=%.2f  U=%.2f  V^{++}=%.2f\n", N, t, U, Vpp)
  println("="^65)
  @printf("  %8s  %14s  %12s  %12s  %10s\n",
    "ΔV", "E", "E/N", "|⟨Sz_tot⟩|", "S(q=0)")
  println("  ", "-"^62)
  vacrit = (0.5 * U) - pi * abs(t)  #approximate value from bosonization
  @show vacrit
  if isnothing(Npart)
    sites = siteinds("Electron", N; conserve_nf=false, conserve_sz=false)
    psi = initial_mps(sites, Npart=N)
  else
    sites = siteinds("Electron", N; conserve_nf=true, conserve_sz=false) #conserve particle number
    psi = initial_mps(sites, Npart=Npart)
  end
  Sq0list = Float64[]
  magzlist = Float64[]

  for dV in dV_range
    Vpm = Vpp + dV
    E, psi, _ = run_dmrg(N, t, U, Vpp, Vpm; sites=sites, initial_psi=psi, verbose=false)
    Sz = expect(psi, "Sz")
    SzSz = correlation_matrix(psi, "Sz", "Sz")
    Sq0 = sum(SzSz) / N
    total_Sz = abs(sum(Sz))
    push!(Sq0list, Sq0)
    push!(magzlist, total_Sz / N)
    @printf("  %8.3f  %14.8f  %12.8f  %12.6f  %10.4f\n",
      dV, E, E / N, total_Sz, Sq0)
  end
  # Plot S(q=0) vs ΔV:
  titlestring = "Predicted ΔV crit = " * @sprintf("%.3f", vacrit) * "\n(N=$N, t=$t, U=$U, V^{++}=$Vpp)"
  if !isnothing(Npart)
    titlestring = titlestring * " Npart=$Npart"
  end
  p = plot(dV_range, Sq0list, marker=:circle, xlabel="ΔV = V^{+-} - V^{++}", ylabel="S(q=0)", title=titlestring, legend=false)
  plot!(dV_range, magzlist, marker=:cross, legend=false)
  display(p)
  savefig(p, "pngfigs/Sq0_vs_dV_N$N" * "_U$U" * "_Vpp" * @sprintf("%.3f", Vpp) * ".png")
end

# ─── Main ────────────────────────────────────────────────────────────────────
function main()
  N = 32
  t = 1.0
  U = 1.0
  Vpp = 1.0    # V^{++}: same-spin NN repulsion
  Vpm = 3.18
  # Vpm = 1.6  # V^{+-}: opposite-spin NN repulsion  (FM: Vpm > Vpp)
  # Vpp = 1e-6
  # Vpm = 1e-6
  Va = Vpp - Vpm
  @show Va
  vacrit = (0.5 * U) - pi * abs(t)  #approximate value from bosonization
  @show vacrit

  println("="^65)
  @printf("N=%d  t=%.2f  U=%.2f  V^{++}=%.2f  V^{+-}=%.2f\n", N, t, U, Vpp, Vpm)
  @printf("ΔV = V^{+-} - V^{++} = %.2f  \n", Vpm - Vpp)
  println("="^65)

  sites_unconstrained = siteinds("Electron", N; conserve_nf=false, conserve_sz=false)
  psi_even = initial_mps(sites_unconstrained, Npart=N)
  psi_odd = initial_mps(sites_unconstrained, Npart=N - 1)

  sites_const = siteinds("Electron", N; conserve_nf=true, conserve_sz=false)


  E_even, psi_even, _ = run_dmrg(N, t, U, Vpp, Vpm; sites=sites_unconstrained, bpin=1E-5, sweepcount=8)
  _, _, neven = measure(psi_even, sites_unconstrained)
  E_odd, psi_odd, _ = run_dmrg(N, t, U, Vpp, Vpm; sites=sites_unconstrained, bpin=1E-5, sweepcount=8)
  _, _, nodd = measure(psi_odd, sites_unconstrained)
  @show E_even, E_odd
  println("Now removed pinning field")
  # psi = E_even < E_odd ? psi_even : psi_odd
  Npart = E_even < E_odd ? Int(round(neven)) : Int(round(nodd))
  @show Npart
  psi = initial_mps(sites_const; Npart)
  E, psi, _ = run_dmrg(N, t, U, Vpp, Vpm; sites=sites_const, initial_psi=psi, bpin=0, sweepcount=100)
  @printf("\nGround state energy:   E   = %.10f\n", E)
  @printf("Energy per site:       E/N = %.10f\n", E / N)
  _, _, _ = measure(psi, sites_const)
  # entanglement_profile(psi)


  # println("-"^20)
  # println("Direct")
  # println("-"^20)
  # Edirect, psidirect, sites_from_direct = run_dmrg(N, t, U, Vpp, Vpm; sites=nothing, initial_psi = nothing, bpin=0, sweepcount=100)
  # _,_ = measure(psidirect, sites_from_direct)
  # @show Edirect, E


  fermicorrmat = correlation_matrix(psi, "Cup", "Cdagup")
  densitycorrmat = correlation_matrix(psi, "Ntot", "Ntot")
  xaxis = 1:N






  p = plot(xaxis[2:end], abs.(fermicorrmat[1, 2:end]), marker=:cicle, label="c^dag_up(1)c_up(x)")
  plot!(xaxis, abs.(densitycorrmat[1, :] .- (0)), marker=:cross, label="n(1)n(x) - <0>")
  plot!(xscale=:log10, yscale=:log10)
  titlestring = "Predicted ΔV crit = " * @sprintf("%.3f", vacrit) * "\n(N=$N, t=$t, U=$U, V^{++}=$Vpp, V^{+-}=$Vpm)"
  plot!(title=titlestring)
  plot!(legend=:outerbottom, legendcolumns=2)
  display(p)

  # Uncomment to scan the FM transition:
  # scan_deltaV(N, t, U, Vpp; dV_range=0.0:0.1:5.0)
end


function scanmain()
    N   = 32
    t   = 1.0
    # idx = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
    idx = 4
    Ulist = (0.1, 0.3, 0.5, 1.0, 3.0, 7.0) 
    U = Ulist[idx]
    # Vpplist = (-2.0, -1.0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 5.0, 7.0)
    Vpplist = (1.0)
    for Vpp in Vpplist
        scan_deltaV(N, t, U, Vpp; dV_range=2.3:0.02:2.4, Npart=14)
    end #for
end

## 
#main()

scanmain()

##
# let 
#   a = [1, 2, 4 , 5]
#   Iterators.product(a, a)
# end
