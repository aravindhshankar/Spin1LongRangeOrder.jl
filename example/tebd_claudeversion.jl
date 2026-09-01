using MKL
using ITensors
using ITensorMPS   # if you're on a recent ITensors split; harmless if not needed
include(joinpath("../calculateFromData", "correlations.jl"))
using Plots
gr()
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

        hj =  -t * op("Cdagup", s1) * op("Cup",    s2)
        hj +=  t * op("Cup",    s1) * op("Cdagup", s2)
        hj += -t * op("Cdagdn", s1) * op("Cdn",    s2)
        hj +=  t * op("Cdn",    s1) * op("Cdagdn", s2)

        hj += Vpp * op("Nup", s1) * op("Nup", s2)
        hj += Vpp * op("Ndn", s1) * op("Ndn", s2)
        hj += Vpm * op("Nup", s1) * op("Ndn", s2)
        hj += Vpm * op("Ndn", s1) * op("Nup", s2)

        fac1 = (j == 1)     ? 1.0 : 0.5
        fac2 = (j == N - 1) ? 1.0 : 0.5
        hj += fac1 * U * op("Nupdn", s1) * op("Id", s2)
        hj += fac2 * U * op("Id",    s1) * op("Nupdn", s2)

        push!(gates, exp(-im * tau / 2 * hj))
    end

    append!(gates, reverse(gates))   # symmetric (2nd order) Trotter step
    return gates
end

# -----------------------------------------------------------------------
# Efficiently compute  f(x) = <psi0| Odag(x) |phi>  for every site x,
# using cumulative left/right overlap environments of <psi0|phi>.
# psi0 and phi must live on the *same* site indices `sites`.
# -----------------------------------------------------------------------
function local_operator_correlation(psi0::MPS, phi::MPS, opdagname::String, sites)
    N = length(psi0)

    L = Vector{ITensor}(undef, N + 1)
    L[1] = ITensor(1.0)
    for j in 1:N
        L[j+1] = L[j] * phi[j] * dag(psi0[j])
    end

    R = Vector{ITensor}(undef, N + 2)
    R[N+1] = ITensor(1.0)
    for j in N:-1:1
        R[j] = R[j+1] * phi[j] * dag(psi0[j])
    end

    Cx = Vector{ComplexF64}(undef, N)
    for x in 1:N
        Odag = op(opdagname, sites[x])
        mid  = phi[x] * Odag * dag(prime(psi0[x], sites[x]))
        Cx[x] = (L[x] * mid * R[x+1])[]
    end
    return Cx
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
)
    # Needed for correct fermion signs when applying local operators/gates
    # by hand (as opposed to letting OpSum build the MPO, which handles
    # Jordan-Wigner strings internally regardless of this flag).
    ITensors.enable_auto_fermion()

    sites = siteinds(psi0)
    N = length(sites)
    x0 = something(x0, N ÷ 2)

    opname, opdagname = _op_names(operator)

    if H === nothing
        H = build_hamiltonian(sites, t, U, Vpp, Vpm)
    end
    if E0 === nothing
        E0 = real(inner(psi0, H, psi0))
    end
    verbose && println("E0 = $E0,  x0 = $x0,  operator = $operator")

    # |phi(0)> = O(x0) |psi0>
    Ox0 = op(opname, sites[x0])
    phi = apply(Ox0, psi0; cutoff=cutoff, maxdim=maxdim)

    # choose an integer number of equal steps hitting tf exactly
    nsteps = max(1, round(Int, tf / dt))
    dt_actual = tf / nsteps
    gates = make_tebd_gates(sites, t, U, Vpp, Vpm, dt_actual)

    for step in 1:nsteps
        phi = apply(gates, phi; cutoff=cutoff, maxdim=maxdim)
        normalize_state && normalize!(phi)
        if verbose
            println("  step $step/$nsteps  (t=$(round(step*dt_actual, digits=4)))  ",
                    "maxlinkdim(phi) = $(maxlinkdim(phi))  norm(phi) = $(round(norm(phi), digits=6))")
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

@time begin
let
# -----------------------------------------------------------------------
# Example usage:
#
N = 16
t, U, Vpp, Vpm = 1.0, 0.1, 0.8, 4.0
filename = filename_builder(N, t, U, Vpp, Vpm)
psi0, params = load_simulation(filename, Val(:all))
println("The initial bond dimension of psi0 is ", ret_maxlinkdim(psi0))
sites = siteinds(psi0)
c = div(N,2)


Cx = dynamical_correlation_tebd(
    psi0, 1.0;
    t=1.0, U=4.0, Vpp=0.5, Vpm=0.5,
    operator = :Splus,          # :charge, :Sz, :Splus, :cup, :cdn
    dt = 0.01,
    x0 = c,
    cutoff = 1e-10,
    maxdim = 600,
    subtract_disconnected = true,
)
initcorrsz = correlation_matrix(psi0, "S+", "S-")[c, :]
xvals = collect(1:length(initcorrsz))
p = plot(xvals, abs.(Cx), label="Dynamic")
plot!(xvals, abs.(initcorrsz), label="Static")
plot!(yscale = :log)
display(p)

# Cx[x] is C(x, t=5.0) for site x = 1:N
# -----------------------------------------------------------------------
end
end