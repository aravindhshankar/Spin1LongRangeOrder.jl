using MKL
using ITensors
using ITensorMPS 
using HDF5

# -----------------------------------------------------------------------
# Checkpoint I/O for resumable TEBD runs.
# -----------------------------------------------------------------------
function save_checkpoint(path::String, phi::MPS, step::Int, t_now::Real, E0::Real)
    tmp = path * ".tmp"   # write-then-rename so a crash mid-write can't corrupt the checkpoint
    h5open(tmp, "w") do f
        write(f, "psi", phi)
        write(f, "step", step)
        write(f, "t_now", t_now)
        write(f, "E0", E0)
    end
    mv(tmp, path; force=true)
end

function load_checkpoint(path::String)
    h5open(path, "r") do f
        phi = read(f, "psi", MPS)
        step = read(f, "step")
        t_now = read(f, "t_now")
        E0 = read(f, "E0")
        return phi, step, t_now, E0
    end
end

# cheap check of how far a checkpoint has already evolved, without loading the MPS
function peek_checkpoint(path::String)
    h5open(path, "r") do f
        return read(f, "step"), read(f, "t_now")
    end
end

# -----------------------------------------------------------------------
# Growing time-series file: one HDF5 group per saved step, so repeated
# calls with increasing tf (using the same checkpoint) keep appending
# snapshots instead of overwriting a single final-time result.
# -----------------------------------------------------------------------
function save_snapshot(path::String, step::Int, t_now::Real, Cx::Vector{ComplexF64})
    h5open(path, "cw") do f
        gname = "step_$step"
        haskey(f, gname) && return
        g = create_group(f, gname)
        write(g, "t", t_now)
        write(g, "Cx_real", real.(Cx))
        write(g, "Cx_imag", imag.(Cx))
    end
end

# current (not peak) resident memory; Linux-only, falls back to peak elsewhere
function current_rss()
    isfile("/proc/self/status") || return Sys.maxrss()
    for l in eachline("/proc/self/status")
        startswith(l, "VmRSS:") && return parse(Int, split(l)[2]) * 1024
    end
    return Sys.maxrss()
end

# -----------------------------------------------------------------------
# Your Hamiltonian, unchanged
# -----------------------------------------------------------------------
function build_hamiltonian(sites, t, U, Vpp, Vpm)
    N  = length(sites)
    os = OpSum()

    for i in 1:(N-1)
        os += -t, "Cdagup", i, "Cup",    i+1
        os +=  t, "Cup",    i, "Cdagup", i+1
        os += -t, "Cdagdn", i, "Cdn",    i+1
        os +=  t, "Cdn",    i, "Cdagdn", i+1

        os += Vpp, "Nup", i, "Nup", i+1
        os += Vpp, "Ndn", i, "Ndn", i+1

        os += Vpm, "Nup", i, "Ndn", i+1
        os += Vpm, "Ndn", i, "Nup", i+1

        os += U, "Nupdn", i
    end
    os += U, "Nupdn", N

    return MPO(os, sites)
end

# -----------------------------------------------------------------------
# Build 2nd-order (Strang) Trotter gates for a single full time step `tau`,
# for exactly the bond/onsite terms appearing in build_hamiltonian above.
# On-site U is split between the two bonds touching a given site (full
# weight only at the chain ends, where a site touches just one bond) so
# that summing over all bonds reproduces H exactly.
# -----------------------------------------------------------------------
function make_tebd_gates(sites, t, U, Vpp, Vpm, tau)
    N = length(sites)
    gates = ITensor[]

    for j in 1:(N - 1)
        s1 = sites[j]
        s2 = sites[j + 1]

        # Built through a 2-site OpSum MPO so the Jordan-Wigner signs are
        # exactly those of build_hamiltonian (the convention DMRG used).
        os = OpSum()
        os += -t, "Cdagup", 1, "Cup",    2
        os +=  t, "Cup",    1, "Cdagup", 2
        os += -t, "Cdagdn", 1, "Cdn",    2
        os +=  t, "Cdn",    1, "Cdagdn", 2

        os += Vpp, "Nup", 1, "Nup", 2
        os += Vpp, "Ndn", 1, "Ndn", 2
        os += Vpm, "Nup", 1, "Ndn", 2
        os += Vpm, "Ndn", 1, "Nup", 2

        fac1 = (j == 1)     ? 1.0 : 0.5
        fac2 = (j == N - 1) ? 1.0 : 0.5
        os += fac1 * U, "Nupdn", 1
        os += fac2 * U, "Nupdn", 2
        hj = prod(MPO(os, [s1, s2]))

        push!(gates, exp(-im * tau / 2 * hj))
    end

    append!(gates, reverse(gates))   # symmetric (2nd order) Trotter step
    return gates
end

# A stepper builds a closure  phi -> phi(t+dt)  once, before the time loop.
# Swapping it (see tdvp_stepper in dynamical_correlation_tdvp.jl) changes
# the time-evolution method without touching the driver below.
function tebd_stepper(phi, H; sites, t, U, Vpp, Vpm, dt, cutoff, maxdim, normalize)
    gates = make_tebd_gates(sites, t, U, Vpp, Vpm, dt)
    return function (phi)
        phi = apply(gates, phi; cutoff=cutoff, maxdim=maxdim)
        normalize && normalize!(phi)
        return phi
    end
end

# -----------------------------------------------------------------------
# Efficiently compute  f(x) = <psi0| Odag(x) |phi>  for every site x,
# using cumulative left/right overlap environments of <psi0|phi>.
# psi0 and phi must live on the *same* site indices `sites`.
# -----------------------------------------------------------------------
function local_operator_correlation(psi0::MPS, phi::MPS, opdagname::String, sites)
    N = length(psi0)
    # Fresh bra link indices: phi can share links with psi0 (always at t=0,
    # since O(x0)|psi0> only touches one site tensor), which would otherwise
    # contract them together.
    bra = dag(sim(linkinds, psi0))
    # fermionic Odag(x) carries a Jordan-Wigner string F on every site < x
    fermionic = ITensors.has_fermion_string(opdagname, sites[1])

    R = Vector{ITensor}(undef, N + 1)
    R[N+1] = ITensor(1.0)
    for j in N:-1:2
        R[j] = (R[j+1] * bra[j]) * phi[j]
    end

    # left environment carried as a single running tensor (only R is stored)
    L = ITensor(1.0)
    Cx = Vector{ComplexF64}(undef, N)
    for x in 1:N
        Lb = L * bra[x]
        Cx[x] = (Lb * noprime(op(opdagname, sites[x]) * phi[x]) * R[x+1])[]
        L = fermionic ? Lb * noprime(op("F", sites[x]) * phi[x]) : Lb * phi[x]
    end
    return Cx
end

# O(x0)|psi>, with the Jordan-Wigner string F on sites < x0 when O is
# fermionic (OpSum refuses parity-odd single-operator MPOs). Single-site
# tensors only, so no links change; truncate afterwards if needed.
function apply_local_operator(psi::MPS, opname::String, x0::Int)
    sites = siteinds(psi)
    phi = copy(psi)
    if ITensors.has_fermion_string(opname, sites[x0])
        for j in 1:(x0 - 1)
            phi[j] = noprime(op("F", sites[j]) * phi[j])
        end
    end
    phi[x0] = noprime(op(opname, sites[x0]) * phi[x0])
    return phi
end

# -----------------------------------------------------------------------
# operator/dagger-operator name pairs.
#   :charge -> local charge density  n(x) = Nup+Ndn   (Hermitian)
#   :Sz     -> local Sz                                (Hermitian)
#   :Splus  -> S^+(x), dagger is S^-(x)
#   :cup    -> c_up(x) annihilation, dagger is Cdagup
#   :cdn    -> c_dn(x) annihilation, dagger is Cdagdn
# -----------------------------------------------------------------------
function _op_names(operator::Symbol)
    operator === :charge && return ("Ntot", "Ntot")
    operator === :Sz     && return ("Sz",   "Sz")
    operator === :Splus  && return ("S+",   "S-")
    operator === :cup    && return ("Cup",  "Cdagup")
    operator === :cdn    && return ("Cdn",  "Cdagdn")
    error("Unknown operator $operator; use :charge, :Sz, :Splus, :cup or :cdn")
end

# -----------------------------------------------------------------------
# Main driver.
#
#   psi0 : ground-state MPS from DMRG (on `sites`)
#   tf   : final time to evolve to
#
# Keyword args control the Hamiltonian (must match the one psi0 was
# obtained from), which operator to use, where the reference operator
# sits (defaults to N÷2), and the TEBD numerics.
#
# Returns a Vector{ComplexF64} of length N: C(x, tf) for x = 1..N.
# -----------------------------------------------------------------------
function dynamical_correlation_tebd(
    psi0::MPS,
    tf::Real;
    t::Real,
    U::Real,
    Vpp::Real,
    Vpm::Real,
    operator::Symbol = :charge,
    x0::Union{Nothing,Int} = nothing,
    dt::Real = 0.02,
    cutoff::Real = 1e-10,
    maxdim::Int = 800,
    normalize_state::Bool = false,
    subtract_disconnected::Bool = false,
    verbose::Bool = true,
    H::Union{Nothing,MPO} = nothing,
    E0::Union{Nothing,Real} = nothing,
    checkpoint_path::Union{Nothing,String} = nothing,
    checkpoint_every::Int = 50,
    snapshot_path::Union{Nothing,String} = nothing,
    snapshot_every::Int = checkpoint_every,
    gc_every::Int = checkpoint_every,
    stepper = tebd_stepper,
)
    # No auto-fermion: the ground states come from standard-mode DMRG, and
    # auto-fermion turns the same OpSum into a different operator. Jordan-
    # Wigner strings are handled explicitly (OpSum MPOs, F in the environment).
    sites = siteinds(psi0)
    N = length(sites)
    x0 = something(x0, N ÷ 2)

    opname, opdagname = _op_names(operator)

    # choose an integer number of equal steps hitting tf exactly
    nsteps = max(1, round(Int, tf / dt))
    dt_actual = tf / nsteps

    start_step = 0
    phi = nothing
    if checkpoint_path !== nothing && isfile(checkpoint_path)
        phi, start_step, t_now, E0 = load_checkpoint(checkpoint_path)
        verbose && println("Resuming from checkpoint: step $start_step/$nsteps (t=$t_now), E0=$E0")
    else
        if H === nothing
            H = build_hamiltonian(sites, t, U, Vpp, Vpm)
        end
        E0 === nothing && (E0 = real(inner(psi0, H, psi0)))
        verbose && println("E0 = $E0,  x0 = $x0,  operator = $operator")
        phi = apply_local_operator(psi0, opname, x0)          # |phi(0)> = O(x0)|psi0>
        truncate!(phi; cutoff=cutoff, maxdim=maxdim)
    end

    advance! = stepper(phi, H; sites=sites, t=t, U=U, Vpp=Vpp, Vpm=Vpm, dt=dt_actual,
                       cutoff=cutoff, maxdim=maxdim, normalize=normalize_state)

    for step in (start_step + 1):nsteps
        phi = advance!(phi)

        if checkpoint_path !== nothing && (step % checkpoint_every == 0 || step == nsteps)
            save_checkpoint(checkpoint_path, phi, step, step * dt_actual, E0)
        end
        if snapshot_path !== nothing && (step % snapshot_every == 0 || step == nsteps)
            t_step = step * dt_actual
            Cx_step = local_operator_correlation(psi0, phi, opdagname, sites)
            Cx_step .*= cis(E0 * t_step)
            save_snapshot(snapshot_path, step, t_step, Cx_step)
        end
        # Julia's GC (and MKL's internal scratch-buffer pool) can let RSS
        # climb well past the live-data size under a cgroup memory limit,
        # since the default GC heuristics are tuned to *system* memory
        # pressure, which a cgroup cap doesn't visibly create until it's
        # too late. Forcing a collection here bounds that drift; combine
        # with MKL_DISABLE_FAST_MM=1 and --heap-size-hint at job submission.
        if step % gc_every == 0 || step == nsteps
            GC.gc()
        end

        if verbose
            rss_gb  = round(current_rss() / 2^30, digits=2)
            live_gb = round(Base.gc_live_bytes() / 2^30, digits=2)
            println("  step $step/$nsteps  (t=$(round(step*dt_actual, digits=4)))  ",
                    "maxlinkdim(phi) = $(maxlinkdim(phi))  norm(phi) = $(round(norm(phi), digits=6))  ",
                    "RSS = $(rss_gb) GB  julia live = $(live_gb) GB")
            flush(stdout)   # cluster logs are only useful live if they're not stuck in a buffer
        end
    end

    # <psi0| Odag(x) |phi(tf)>  for every x
    Cx = local_operator_correlation(psi0, phi, opdagname, sites)

    # multiply by the eigenstate phase e^{i E0 tf}
    Cx .*= cis(E0 * tf)

    if subtract_disconnected
        operator in (:charge, :Sz) ||
            @warn "subtract_disconnected requested for a non-Hermitian operator; " *
                  "the disconnected piece is normally zero by symmetry in that case."
        Ox_static = expect(psi0, opname)         # <O(x)> for all x
        Cx .-= Ox_static .* Ox_static[x0]
    end

    return Cx
end

# -----------------------------------------------------------------------
# Example usage. Run to tf=1 first, look at the snapshots, then call again
# with a bigger tf to keep going -- it resumes from checkpoint_path rather
# than restarting, and snapshot_path accumulates every saved time (not just
# the final one).
#
# psi0 = load_simulation(filepath)
#
# dynamical_correlation_tebd(
#     psi0, 1.0;
#     t=1.0, U=4.0, Vpp=0.5, Vpm=0.5, operator=:charge,
#     dt=0.01, cutoff=1e-9, maxdim=1000,
#     checkpoint_path="ckpt.h5", snapshot_path="snapshots.h5", snapshot_every=10,
# )
# # ... inspect snapshots.h5, decide it's not enough ...
# dynamical_correlation_tebd(
#     psi0, 5.0;
#     t=1.0, U=4.0, Vpp=0.5, Vpm=0.5, operator=:charge,
#     dt=0.01, cutoff=1e-9, maxdim=1000,
#     checkpoint_path="ckpt.h5", snapshot_path="snapshots.h5", snapshot_every=10,
# )
# -----------------------------------------------------------------------
