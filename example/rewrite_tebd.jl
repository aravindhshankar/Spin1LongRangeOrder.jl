using MKL
using Spin1LongRangeOrder
using ITensors, ITensorMPS
include(joinpath("../calculateFromData", "correlations.jl"))
using Plots
gr()

##
# -----------------------------------------------------------------------
# Bond Hamiltonian with the on-site U folded in (full weight at chain
# ends where a site touches only one bond, half weight in the bulk where
# a site touches two bonds), so that summing h_i over ALL bonds i=1..N-1
# reproduces the full Hamiltonian exactly. This lets us drop the separate
# onsite gate layer entirely and use a pure even/odd checkerboard Trotter.
# -----------------------------------------------------------------------
function bond_hamiltonian_with_onsite(sites, i, t, U, Vpp, Vpm, N)
    s1 = sites[i]
    s2 = sites[i + 1]

    hj  = -t * op("Cdagup", s1) * op("Cup",    s2)
    hj +=  t * op("Cup",    s1) * op("Cdagup", s2)
    hj += -t * op("Cdagdn", s1) * op("Cdn",    s2)
    hj +=  t * op("Cdn",    s1) * op("Cdagdn", s2)

    hj += Vpp * op("Nup", s1) * op("Nup", s2)
    hj += Vpp * op("Ndn", s1) * op("Ndn", s2)
    hj += Vpm * op("Nup", s1) * op("Ndn", s2)
    hj += Vpm * op("Ndn", s1) * op("Nup", s2)

    fac1 = (i == 1)     ? 1.0 : 0.5
    fac2 = (i == N - 1) ? 1.0 : 0.5
    hj += fac1 * U * op("Nupdn", s1) * op("Id", s2)
    hj += fac2 * U * op("Id",    s1) * op("Nupdn", s2)

    return hj
end

# -----------------------------------------------------------------------
# Even/odd (checkerboard) 2nd-order Trotter gates for one full step of
# size dt. Odd bonds mutually commute, even bonds mutually commute, so
# each group's exponential is a plain product with no extra Trotter
# error inside the group.
#
#   odd_half  : exp(-i h_odd  dt/2)   -- used at the very start/end
#   odd_full  : exp(-i h_odd  dt)     -- used to merge two half-steps
#                                          from consecutive time steps
#   even_full : exp(-i h_even dt)
# -----------------------------------------------------------------------
function build_checkerboard_gates(sites, t, U, Vpp, Vpm, dt)
    N = length(sites)
    odd_bonds  = 1:2:(N - 1)
    even_bonds = 2:2:(N - 1)

    hbond(i) = bond_hamiltonian_with_onsite(sites, i, t, U, Vpp, Vpm, N)

    odd_half  = [exp(-im * dt / 2 * hbond(i)) for i in odd_bonds]
    odd_full  = [exp(-im * dt     * hbond(i)) for i in odd_bonds]
    even_full = [exp(-im * dt     * hbond(i)) for i in even_bonds]

    return odd_half, odd_full, even_full
end

# -----------------------------------------------------------------------
# Evolve psi by exactly ttotal using nsteps checkerboard Trotter steps,
# with the boundary half-steps between consecutive steps merged into
# full-weight applications (saves ~1/3 of the expensive 2-site gate
# applications compared to naively repeating [half,full,half] each step).
# -----------------------------------------------------------------------
function tebd_evolve!(psi, sites, t, U, Vpp, Vpm, ttotal, tau; cutoff, maxdim, verbose=true)
    nsteps = max(1, round(Int, ttotal / tau))
    dt = ttotal / nsteps   # land exactly on ttotal even if ttotal isn't an exact multiple of tau

    odd_half, odd_full, even_full = build_checkerboard_gates(sites, t, U, Vpp, Vpm, dt)

    psi = apply(odd_half, psi; cutoff, maxdim)
    psi = apply(even_full, psi; cutoff, maxdim)
    for step in 2:nsteps
        psi = apply(odd_full, psi; cutoff, maxdim)
        psi = apply(even_full, psi; cutoff, maxdim)
        if verbose
            println("  step $step/$nsteps  t=$(round(step*dt, digits=6))  ",
                    "maxlinkdim=$(maxlinkdim(psi))  norm=$(round(norm(psi), digits=6))")
        end
    end
    psi = apply(odd_half, psi; cutoff, maxdim)

    return psi
end

# -----------------------------------------------------------------------
# Efficient O(N χ³) computation of f(x) = <bra| Odag(x) |ket> for every
# site x at once, via cumulative left/right overlap environments of
# <bra|ket>. Replaces the O(N² χ³) loop of apply()+inner() per site.
# bra and ket must share the same site indices `sites`.
# -----------------------------------------------------------------------
function local_operator_correlation(bra::MPS, ket::MPS, opdagname::String, sites)
    N = length(bra)

    L = Vector{ITensor}(undef, N + 1)
    L[1] = ITensor(1.0)
    for j in 1:N
        L[j+1] = L[j] * ket[j] * dag(bra[j])
    end

    R = Vector{ITensor}(undef, N + 2)
    R[N+1] = ITensor(1.0)
    for j in N:-1:1
        R[j] = R[j+1] * ket[j] * dag(bra[j])
    end

    Cx = Vector{ComplexF64}(undef, N)
    for x in 1:N
        Odag = op(opdagname, sites[x])
        mid  = ket[x] * Odag * dag(prime(bra[x], sites[x]))
        Cx[x] = (L[x] * mid * R[x+1])[]
    end
    return Cx
end

@time begin
let
    N = 16
    t, U, Vpp, Vpm = 1.0, 0.1, 0.8, 4.0
    filename = filename_builder(N, t, U, Vpp, Vpm)
    psi0, params = load_simulation(filename, Val(:all))
    println("The initial bond dimension of psi0 is ", ret_maxlinkdim(psi0))

    # ITensors.enable_auto_fermion()

    sites = siteinds(psi0)
    tau = 1E-2
    ttotal = 100 * tau        # bump this up once you've validated correctness

    cutoff = 1e-12
    maxdim = 600

    psi = copy(psi0)
    c = div(N, 2)
    opname = "S+"
    opdagname = "S-"

    psi = apply(op(opname, sites[c]), psi; cutoff, maxdim)  # perturb init state: |phi(0)> = O(c)|psi0>
    psi = tebd_evolve!(psi, sites, t, U, Vpp, Vpm, ttotal, tau; cutoff, maxdim)
    println("The final bond dimension is ", ret_maxlinkdim(psi))

    # <psi0| Odag(x) |phi(t)>  for every x, in one O(Nχ³) sweep
    corrsz = local_operator_correlation(psi0, psi, opdagname, sites)

    # NOTE: this is missing the overall phase e^{i E0 t} relative to the
    # true C(x,t) = <psi0|Odag(x,t)O(c,0)|psi0>. Since psi0 is (approx.)
    # an eigenstate, that phase is a *global* constant, so it does not
    # affect abs.(corrsz) below. If you later want the complex/real/imag
    # parts (e.g. to Fourier transform to S(x,ω)), compute
    #   H = build_hamiltonian(sites, t, U, Vpp, Vpm)   # your original fn
    #   E0 = real(inner(psi0, H, psi0))
    # and multiply corrsz by cis(E0*ttotal).

    # static (t=0) reference, computed with the SAME operator ordering as
    # the dynamic correlator above so the two are directly comparable
    # phi0 = apply(op(opname, sites[c]), psi0; cutoff, maxdim)
    # initcorrsz = local_operator_correlation(psi0, phi0, opdagname, sites)
    initcorrsz = correlation_matrix(psi0, opname, opdagname)[c, :]
    xvals = collect(1:length(corrsz))
    p = plot(xvals, abs.(corrsz), label="Dynamic")
    plot!(xvals, abs.(initcorrsz), label="Static")
    plot!(yscale = :log)
    display(p)
end
end