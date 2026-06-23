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

# ─── Sweep schedule ───────────────────────────────────────────────────────────
function make_sweeps(max_sweeps=50)
    # Each entry: (maxdim, cutoff, noise)
    # The schedule has 50 rows; bond dim and noise saturate after row 6.
    base = [
        ("maxdim", "cutoff", "noise"),  
        fill((20,   1e-6,  1e-3), 2)...,
        fill((50,   1e-8,  1e-4), 3)...,
        fill((100,  1e-10, 1e-5), 5)...,
        fill((200,  1e-12,  0.0), 10)...,
        fill((400,  1e-12, 0.0), 10)...,
        (500,  1e-12, 0.0),
    ]
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
function initial_mps(sites; Npart=nothing)
    N     = length(sites)
    # Npart = isnothing(Npart) ? length(sites) : Npart
    #force a specific Npart, break without error if not
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
    psi0 = random_mps(sites, state; linkdims = 10)

    # Check total number of particles:
    # @show flux(Spsi0)
    return psi0
end

# ─── Run a single DMRG calculation ───────────────────────────────────────────
# WARNING! DON'T RUN WITHOUT USING THE SCAN_DELTAV 
function run_dmrg(N, t, U, Vpp, Vpm; sites=nothing, initial_psi=nothing, verbose=true)
    sites = isnothing(sites) ? siteinds("Electron", N; conserve_nf=true, conserve_sz=false) : sites
    # sites = siteinds("Electron", N; conserve_qns=true)
    H     = build_hamiltonian(sites, t, U, Vpp, Vpm)
    psi0  = isnothing(initial_psi) ? initial_mps(sites) : initial_psi
    sw    = make_sweeps(150)

    obs    = EnergyObserver(energy_tol=1e-8, min_sweeps=25)
    eigsolve_krylovdim = 10
    @time E, psi = dmrg(H, psi0, sw; eigsolve_krylovdim, outputlevel=verbose ? 1 : 0, observer=obs)
    var =   variance_gs(H, psi)
    @printf("\nGround state energy variance:  = %.2e\n", var)
    print("Total electron density: ⟨N⟩ = ")
    number_profile = expect(psi, "Ntot")
    # @show number_profile
    total_N = sum(number_profile)
    @printf("%.4f  (expected %d)\n", total_N, N)
    return E, psi, sites, var
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
    Sz_per_spin = 0.5 * sum(Nup_v - Ndn_v ./ Ntot)  
    @show Sz_per_spin
    total_N     = sum(Ntot)
    mean_double = sum(double) / N
    total_Sz_per_spin = total_Sz/total_N

    label == "" || println("\n── $label ──")
    @printf("  Total ⟨Sz⟩     = %+.6f\n", total_Sz)
    @printf("  Total ⟨Sz⟩ per spin     = %+.6f\n", total_Sz_per_spin)
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
    # println("\n  Spin structure factor S(q):")
    # for q in range(0, π, length=9)
    #     Sq = sum(cos(q*(i-j)) * SzSz[i,j] for i in 1:N, j in 1:N) / N
    #     @printf("    q/π = %.3f:  S(q) = %+.6f\n", q/π, Sq)
    # end

    if abs(total_Sz) > 0.3 * N / 2
        println("\n  >>> Strong ferromagnetic order detected!")
    elseif maximum(abs.(vec(sum(SzSz, dims=2)))) > 2.0
        println("\n  >>> Large uniform spin correlations — near FM critical point.")
    else
        println("\n  >>> No ferromagnetic order at these parameters.")
    end

    return Sz, SzSz
end



# ─── Scan ΔV = V^{+-} - V^{++} to locate the FM transition ──────────────────
function scan_deltaV(N=16, t=1.0, U=4.0, Vpp=0.5, Npart=7;
                     dV_range=0.0:0.25:2.0, startmps=nothing)
    println("\n", "="^65)
    println("Scanning ΔV = V^{+-} - V^{++} to locate FM transition")
    println("Fixed Npart = ", Npart)
    @printf("  N=%d  t=%.2f  U=%.2f  V^{++}=%.2f\n", N, t, U, Vpp)
    println("="^65)
    @printf("  %8s  %14s  %12s  %12s  %10s\n",
            "ΔV", "E", "E/N", "|⟨Sz_tot⟩|", "S(q=0)")
    println("  ", "-"^62)
    vacrit = (0.5 * U) - pi * abs(t)  #approximate value from bosonization
    @show vacrit
    
    if isnothing(startmps)
        psi = initial_mps(sites, Npart=Npart)
        sites = siteinds("Electron", N; conserve_nf=true, conserve_sz=false)
    else
        psi = startmps
        sites = siteinds(psi)
    end

    Sq0list = Float64[]
    magzlist = Float64[]
    datasavedir = "data/Hubbard/N$N" * "consNf/"
    mkpath(datasavedir)

    for dV in dV_range
        Vpm              = Vpp + dV
        E, psi, _, var   = run_dmrg(N, t, U, Vpp, Vpm; sites=sites, initial_psi=psi, verbose=false)
        Sz               = expect(psi, "Sz")
        SzSz             = correlation_matrix(psi, "Sz", "Sz")
        Sq0              = sum(SzSz) / N
        total_Sz         = abs(sum(Sz))
        total_szbyN      = total_Sz / N
        max_linkdim_psi  = ret_maxlinkdim(psi)
        push!(Sq0list, Sq0)
        push!(magzlist, total_Sz/N)
        @printf("  %8.3f  %14.8f  %12.8f  %12.6f  %10.4f\n",
                dV, E, E/N, total_Sz, Sq0)
        @show max_linkdim_psi
        datafilename = datasavedir * "N$N" * "_U" * @sprintf("%.3f", U) * "_Vpp" * @sprintf("%.3f", Vpp) * "_Vpm" * @sprintf("%.3f", Vpm) * "_Np$Npart" * raw".h5"
        hamparams = (t=t, U=U, Vpp=Vpp, Vpm=Vpm, dV=dV, totalSz=total_szbyN, Sqzero=Sq0, maxlinkdim=max_linkdim_psi, Npart=Npart, E=E, var=var, conserve=Npart)
    
        save_simulation(datafilename, psi, Dict(pairs(hamparams)))
        flush(stdout)
    end
    # Plot S(q=0) vs ΔV:
    titlestring = "Predicted ΔV crit = " * @sprintf("%.3f", vacrit) * "\n(N=$N, t=$t, U=$U, V^{++}=$Vpp, Npart=$Npart)"
    p = plot(dV_range, Sq0list, marker=:circle, xlabel="ΔV = V^{+-} - V^{++}", ylabel="S(q=0)", title=titlestring, legend=false)
    plot!(dV_range, magzlist, marker=:cross,legend=false)
    # display(p)
    savedir = "pngfigs/Hubbard/N$N" * "fixed/"
    mkpath(savedir) #tested that it works
    savefilename = savedir * "scan$N"*"Npart$Npart"*"_U"*@sprintf("%.3f", U)*"_Vpp" * @sprintf("%.3f", Vpp) * ".png"
    savefig(p, savefilename)
end

##
function filename_builder(N, t, U, Vpp, dV)
    Vpm = Vpp + dV
    Npart = Int(N // 2)
    _ = t # not used, so we discard, but leave the API as is for the future
    datasavedir = "data/Hubbard/N$N" * "consNf/"
    datafilename = datasavedir * "N$N" * "_U" * @sprintf("%.3f", U) * "_Vpp" * @sprintf("%.3f", Vpp) * "_Vpm" * @sprintf("%.3f", Vpm) * "_Np$Npart" * raw".h5"
    return datafilename
end
##

# ─── Main ────────────────────────────────────────────────────────────────────
function main()
    N   = 64
    t   = 1.0
    # idx = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
    # total_tasks = Base.parse(Int, ENV["SLURM_ARRAY_TASK_COUNT"]) # assumes 1-based indexing
    # idx = 1
    U = 0.1
    Vpp = 0.8
    # Npartlist = 5:30
    # this_job_nparts = get_chunk(Npartlist, idx, total_tasks)
    this_job_nparts = (Int(N//2),)

    Vpmreloadict = Dict(((64, 4.500), (128, 4.100), (256, 3.950))) # last available .h5 on disk
    Vpmreloadval = Vpmreloadict(N)
    dvreloadval = round(Vpmreloadval - Vpp, digits=3)
    loadfilename = filename_builder(N, t, U, Vpp, dvreloadval)

    println("-"^20, "loading from file", loadfilename, "-"^20)
    startmps = load_simulation(loadfilename)
    println("Load successful! Resuming simulation ...")
    flush(stdout)

    step = 0.05
    start_dv_val = dvreloadval + step
    end_dv_val = 4.2
    # default dV vals : 3.2:0.05:4.2
    
    for Npart in this_job_nparts
        scan_deltaV(N, t, U, Vpp, Npart; dV_range=start_dv_val:step:end_dv_val, startmps)
        # scan_deltaV(N, t, U, Vpp, Npart; dV_range=3.3:-0.01:3.0)
    end #for
end
## 
let 
    main()
end
