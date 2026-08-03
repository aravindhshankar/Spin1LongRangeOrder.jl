using MKL
using Spin1LongRangeOrder
using ITensors, ITensorMPS
using HDF5
using LinearAlgebra
using Printf

function make_sweeps(max_sweeps=50)
    # Each entry: (maxdim, cutoff, noise)
    base = [
        ("maxdim", "cutoff", "noise"),  
        fill((20,   1e-6,  1e-3), 4)...,
        fill((30,   1e-7,  1e-4), 5)...,
        fill((50,   1e-8,  1e-5), 6)...,
        fill((100,  1e-10, 1e-8), 6)...,
        fill((200,  1e-12, 1e-12), 5)...,
        fill((400,  1e-12, 0.0), 10)...,
        (500,  1e-12, 0.0),
    ]
    sw = Sweeps(max_sweeps, stack(base, dims=1))
    return sw
end
##
function make_restart_sweeps(initial_maxdim; max_sweeps=30)

    target = min(ceil(Int, 1.5 * initial_maxdim), 1200)

    # Smooth ramp in bond dimension
    d2 = round(Int, initial_maxdim + (target - initial_maxdim) / 3)
    d3 = round(Int, initial_maxdim + 2 * (target - initial_maxdim) / 3)

    schedule = [
        ("maxdim", "cutoff", "noise"),

        fill((initial_maxdim, 1e-9, 1e-3), 3)...,
        fill((d2,             1e-10, 1e-4), 3)...,
        fill((d3,             1e-10, 1e-6), 2)...,
        fill((target,         1e-11, 1e-8), 2)...,
        fill((target,         1e-11, 0.0), max(5, max_sweeps - 18))...,
    ]

    schedule = schedule[1:min(end, max_sweeps + 1)]

    return Sweeps(max_sweeps, stack(schedule, dims=1))
end
##
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

##
# ─── Build MPO ────────────────────────────────────────────────────────────────
function build_hamiltonian(sites, t, U, Vpp, Vpm, hzbdy=0.0, hzbulk=0.0)
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

        #hz bulk field
        os += hzbulk, "Sz", i
    end

    os += U, "Nupdn", N  # On-site U for the last site
    os += hzbulk, "Sz", N  # Bulk field for the last site
    os += hzbdy, "Sz", N  # Boundary field for the last site
    os += hzbdy, "Sz", 1  # Boundary field for the first site

    return MPO(os, sites)
end
##
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
# WARNING! careful while using the sites object
function run_dmrg(N, t, U, Vpp, Vpm, hzbdy=0.0, hzbulk=0.0; init_dim=nothing, sites=nothing, initial_psi=nothing, verbose=true)
    sites = isnothing(sites) ? siteinds("Electron", N; conserve_nf=true, conserve_sz=false) : sites
    # sites = siteinds("Electron", N; conserve_qns=true)
    H     = build_hamiltonian(sites, t, U, Vpp, Vpm, hzbdy, hzbulk)
    psi0  = isnothing(initial_psi) ? initial_mps(sites) : initial_psi
    # sw    = make_sweeps(50)
    sw = make_restart_sweeps(init_dim; max_sweeps=30)

    obs    = EnergyObserver(energy_tol=1e-5, min_sweeps=10)
    eigsolve_krylovdim = 10
    @time E, psi = dmrg(H, psi0, sw; eigsolve_krylovdim, outputlevel=verbose ? 1 : 0, observer=obs)
    var =   variance_gs(H, psi)
    @printf("\nGround state energy variance:  = %.2e\n", var)
    print("Total electron density: ⟨N⟩ = ")
    number_profile = expect(psi, "Ntot")
    # @show number_profile
    total_N = sum(number_profile)
    @printf("%.4f\n", total_N)
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


##
function filename_builder(N, t, U, Vpp, dV; prefix="data/Hubbard/", makepath=false)
    Vpm = Vpp + dV
    Npart = Int(N // 2)
    _ = t # not used, so we discard, but leave the API as is for the future
    datasavedir = joinpath(prefix, "N$N", "consNf/")
    makepath && mkpath(datasavedir)
    datafilename = datasavedir * "N$N" * "_U" * @sprintf("%.3f", U) * "_Vpp" * @sprintf("%.3f", Vpp) * "_Vpm" * @sprintf("%.3f", Vpm) * "_Np$Npart" * raw".h5"
    return datafilename
end
##
# ─── Main ────────────────────────────────────────────────────────────────────
function main()
    Nlist = (128, 256)
    idx = (haskey(ENV, "SLURM_ARRAY_TASK_ID") ? parse(Int, ENV["SLURM_ARRAY_TASK_ID"]) : 1)
    total_tasks = (haskey(ENV, "SLURM_ARRAY_TASK_COUNT") ? parse(Int, ENV["SLURM_ARRAY_TASK_COUNT"]) : 1)
    t   = 1.0
    U = 0.1
    Vpp = 0.8
    Vpm, linkdim_init, Npart, E_init, var_init  = nothing, nothing, nothing, nothing, nothing
    # allfilenames = collect(Iterators.flatten([joinpath.("data/Hubbard/N$nval" * "consNf/", readdir("data/Hubbard/N$nval" * "consNf/")) for nval in Nlist]))
    allfilenames = String[]
    for N in Nlist 
        for dV in 3.4:0.05:3.8
            this_file = filename_builder(N, t, U, Vpp, dV)
            if isfile(this_file)
                push!(allfilenames, this_file)
            end
        end
    end
    this_job_filenames = get_chunk(allfilenames, idx, total_tasks)
    @show this_job_filenames

    for datafilename in this_job_filenames[1:1]
        println("\n" * "="^80)
        println("Processing file: $datafilename")
        psi, params = nothing, nothing 
        try 
            psi , params = load_simulation(datafilename, Val(:params)) 
            @printf("Loaded data from %s\n", datafilename)
        catch e 
            @warn "Failed to load $datafilename: $e"
            continue
        end
        try 
            t, U, Vpp, Vpm = params["t"], params["U"], params["Vpp"], params["Vpm"]
            linkdim_init = params["maxlinkdim"]
            Npart = params["Npart"]
            E_init = params["E"]
            var_init = params["var"]
            @show var_init, linkdim_init, E_init
        catch e 
            @warn "Failed to extract parameters from $datafilename: $e"
            continue
        end
        N = length(psi)
        hzbdy = 0.0
        hzbulk = 1e-10
        E, psi, _, var = run_dmrg(N, t, U, Vpp, Vpm, hzbdy, hzbulk; init_dim=linkdim_init, sites=siteinds(psi), initial_psi=psi, verbose=true)
        println("New energy: $E, variance: $var")
        if E < E_init && var < var_init
            println("Vpm = $Vpm")
            println("New energy and variance are lower than previous. Saving updated data.")
            params["maxlinkdim"] = ret_maxlinkdim(psi)
            params["E"] = E
            params["var"] = var
            Sz               = expect(psi, "Sz")
            SzSz             = correlation_matrix(psi, "Sz", "Sz")
            Sq0              = sum(SzSz) / N
            total_Sz         = abs(sum(Sz))
            # total_szbyN      = total_Sz / N
            params["totalSz"] = total_Sz
            params["Sqzero"]  = Sq0
            savefilename = filename_builder(N, t, U, Vpp, Vpm; prefix="data/Hubbard/ImpPrec/", makepath=true)
            save_simulation(savefilename, psi, params)
            println("Saved new MPS to filename : $savefilename")
        else
            println("No improvement in energy AND variance. Not saving.")
        end
        # hamparams = (t=t, U=U, Vpp=Vpp, Vpm=Vpm, dV=dV, totalSz=total_szbyN, Sqzero=Sq0, maxlinkdim=max_linkdim_psi, Npart=Npart, E=E, var=var, conserve=Npart)
    end

    flush(stdout)

end
## 
let 
    main()
end
