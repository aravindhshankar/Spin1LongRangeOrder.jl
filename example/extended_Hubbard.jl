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
# Up to 50 sweeps, ramping bond dimension and decreasing noise.
# Early stopping is handled by EnergyObserver below.
function make_sweeps()
    # Each entry: (maxdim, cutoff, noise)
    # The schedule has 50 rows; bond dim and noise saturate after row 6.
    base = [
        ("maxdim", "cutoff", "noise"),  
        fill((20,   1e-6,  1e-4), 2)...,
        fill((50,   1e-8,  1e-6), 2)...,
        fill((100,  1e-10, 1e-8), 5)...,
        (200,  1e-10,  0.0),
        (400,  1e-10, 1e-10),
        (800,  1e-11, 0.0),
    ]
    max_sweeps = 50
    sw = Sweeps(max_sweeps, stack(base, dims=1))
    return sw
end

# ─── Early-stopping observer ──────────────────────────────────────────────────
# Halts DMRG when |ΔE| < energy_tol for one full sweep, after min_sweeps.
mutable struct EnergyObserver <: AbstractObserver
    energy_tol  :: Float64
    min_sweeps  :: Int
    last_energy :: Float64
    sweep_count :: Int
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
function build_hamiltonian(sites, t, U, Vpp, Vpm)
    N  = length(sites)
    os = OpSum()

    for i in 1:(N-1)
        # NN hopping (both spins)
        os += -t, "Cdagup", i, "Cup",    i+1
        os +=  t, "Cup",    i, "Cdagup", i+1   # h.c.
        os += -t, "Cdagdn", i, "Cdn",    i+1
        os +=  t, "Cdn",    i, "Cdagdn", i+1   # h.c.

        # V^{++}: same-spin NN repulsion (↑↑ and ↓↓)
        os += Vpp, "Nup", i, "Nup", i+1
        os += Vpp, "Ndn", i, "Ndn", i+1

        # V^{+-}: opposite-spin NN repulsion (↑↓ and ↓↑)
        os += Vpm, "Nup", i, "Ndn", i+1
        os += Vpm, "Ndn", i, "Nup", i+1

        # On-site Hubbard U
        os += U, "Nupdn", i
    end

    os += U, "Nupdn", N  # On-site U for the last site

    return MPO(os, sites)
end

# ─── Initial MPS (alternating ↑↓, half filling) ──────────────────────────────
function initial_mps(sites)
    N     = length(sites)
    # state = [isodd(i) ? "Up" : "Dn" for i in 1:N]
    # return MPS(sites, state)
    Npart = length(sites) 
    state = ["Emp" for n in 1:N]
    p = Npart
    for i in N:-1:1
        if p > i
            println("Doubly occupying site $i")
            state[i] = "UpDn"
            p -= 2
        elseif p > 0
            println("Singly occupying site $i")
            state[i] = (isodd(i) ? "Up" : "Dn")
            p -= 1
        end
    end
    # Initialize wavefunction to be bond
    # dimension 10 random MPS with number
    # of particles the same as `state`
    psi0 = random_mps(sites, state; linkdims = 10)

    # Check total number of particles:
    @show flux(psi0)
    return psi0
end

# ─── Run a single DMRG calculation ───────────────────────────────────────────
function run_dmrg(N, t, U, Vpp, Vpm; verbose=true)
    sites = siteinds("Electron", N; conserve_nf=true, conserve_sz=false)
    # sites = siteinds("Electron", N; conserve_qns=true)
    H     = build_hamiltonian(sites, t, U, Vpp, Vpm)
    psi0  = initial_mps(sites)
    sw    = make_sweeps()

    obs    = EnergyObserver(energy_tol=1e-4, min_sweeps=3)
    E, psi = dmrg(H, psi0, sw; outputlevel=verbose ? 1 : 0, observer=obs)
    var =   variance_gs(H, psi)
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
    N      = length(sites)
    Sz     = expect(psi, "Sz")
    Nup_v  = expect(psi, "Nup")
    Ndn_v  = expect(psi, "Ndn")
    Ntot   = expect(psi, "Ntot")
    double = expect(psi, "Nupdn")

    total_Sz    = sum(Sz)
    total_N     = sum(Ntot)
    mean_double = sum(double) / N

    label == "" || println("\n── $label ──")
    @printf("  Total ⟨Sz⟩     = %+.6f\n", total_Sz)
    @printf("  Total ⟨N⟩      = %.4f  (expected %d)\n", total_N, N)
    @printf("  Mean ⟨n↑n↓⟩    = %.6f  (double occupancy)\n", mean_double)

    println("\n  Site-resolved observables:")
    println("  ", "-"^55)
    @printf("  %4s  %8s  %8s  %8s  %8s\n", "site", "⟨Sz⟩", "⟨n↑⟩", "⟨n↓⟩", "⟨ntot⟩")
    println("  ", "-"^55)
    for i in 1:N
        @printf("  %4d  %+8.5f  %8.5f  %8.5f  %8.5f\n",
                i, Sz[i], Nup_v[i], Ndn_v[i], Ntot[i])
    end

    # Spin-spin correlations from the central site
    mid  = N ÷ 2
    SzSz = correlation_matrix(psi, "Sz", "Sz")
    println("\n  ⟨Sz_$(mid) Sz_j⟩ spin correlations:")
    for j in 1:N
        @printf("    j=%2d: %+.6f\n", j, SzSz[mid, j])
    end

    # Spin structure factor S(q) = (1/N) Σ_{ij} cos(q(i-j)) ⟨Sz_i Sz_j⟩
    println("\n  Spin structure factor S(q):")
    for q in range(0, π, length=9)
        Sq = sum(cos(q*(i-j)) * SzSz[i,j] for i in 1:N, j in 1:N) / N
        @printf("    q/π = %.3f:  S(q) = %+.6f\n", q/π, Sq)
    end

    if abs(total_Sz) > 0.3 * N / 2
        println("\n  >>> Strong ferromagnetic order detected!")
    elseif maximum(abs.(vec(sum(SzSz, dims=2)))) > 2.0
        println("\n  >>> Large uniform spin correlations — near FM critical point.")
    else
        println("\n  >>> No ferromagnetic order at these parameters.")
    end

    return Sz, SzSz
end

""" Compute the von Neumann entanglement entropy S = -Tr(ρ log ρ) across bond b. """
function entanglement_entropy(psi, b)
  psi = orthogonalize(psi, b)
  U,S,V = svd(psi[b], (linkinds(psi, b-1)..., siteinds(psi, b)...))
  SvN = 0.0
  for n=1:dim(S, 1)
    p = S[n,n]^2
    SvN -= p * log(p)
  end
return SvN
end


# ─── Entanglement entropy profile ────────────────────────────────────────────
function entanglement_profile(psi)
    N = length(psi)
    println("\n  Entanglement entropy S(bond):")
    for b in 1:(N-1)
        s   = entanglement_entropy(psi, b)
        bar = repeat("█", round(Int, s * 10))
        @printf("  bond %2d–%2d:  S = %.4f  %s\n", b, b+1, s, bar)
    end
end

# ─── Scan ΔV = V^{+-} - V^{++} to locate the FM transition ──────────────────
function scan_deltaV(N=20, t=1.0, U=4.0, Vpp=0.5;
                     dV_range=0.0:0.25:2.0)
    println("\n", "="^65)
    println("Scanning ΔV = V^{+-} - V^{++} to locate FM transition")
    @printf("  N=%d  t=%.2f  U=%.2f  V^{++}=%.2f\n", N, t, U, Vpp)
    println("="^65)
    @printf("  %8s  %14s  %12s  %12s  %10s\n",
            "ΔV", "E", "E/N", "|⟨Sz_tot⟩|", "S(q=0)")
    println("  ", "-"^62)

    for dV in dV_range
        Vpm          = Vpp + dV
        E, psi, _   = run_dmrg(N, t, U, Vpp, Vpm; verbose=false)
        Sz           = expect(psi, "Sz")
        SzSz         = correlation_matrix(psi, "Sz", "Sz")
        Sq0          = sum(SzSz) / N
        total_Sz     = abs(sum(Sz))
        @printf("  %8.3f  %14.8f  %12.8f  %12.6f  %10.4f\n",
                dV, E, E/N, total_Sz, Sq0)
    end
end

# ─── Main ────────────────────────────────────────────────────────────────────
function main()
    N   = 8
    t   = 1.0
    U   = 4.0
    Vpp = 0.5    # V^{++}: same-spin NN repulsion
    Vpm = 1.5    # V^{+-}: opposite-spin NN repulsion  (FM: Vpm > Vpp)
    # Vpp = 1e-6
    # Vpm = 1e-6

    println("="^65)
    println("1D Hubbard + spin-dependent NN repulsion (Kun Yang 2004)")
    println("FM transition via V^{+-} > V^{++} mechanism")
    @printf("N=%d  t=%.2f  U=%.2f  V^{++}=%.2f  V^{+-}=%.2f\n", N, t, U, Vpp, Vpm)
    @printf("ΔV = V^{+-} - V^{++} = %.2f  (> 0 drives FM)\n", Vpm - Vpp)
    println("="^65)

    E, psi, sites = run_dmrg(N, t, U, Vpp, Vpm)

    @printf("\nGround state energy:   E   = %.10f\n", E)
    @printf("Energy per site:       E/N = %.10f\n", E/N)

    measure(psi, sites)
    entanglement_profile(psi)

    # Uncomment to scan the FM transition:
    # scan_deltaV(N, t, U, Vpp)
end
## 
main()


