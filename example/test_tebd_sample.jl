using Spin1LongRangeOrder
using ITensors, ITensorMPS
include(joinpath("../calculateFromData", "correlations.jl")) 

##
# ================================================================
# Two-site Hamiltonian h_{i,i+1}
# ================================================================

function bond_hamiltonian(sites, i, t, Vpp, Vpm)
    s1 = sites[i]
    s2 = sites[i + 1]
    hj =
        -t * op("Cdagup", s1) * op("Cup", s2) +
         t * op("Cup",    s1) * op("Cdagup", s2) +
        -t * op("Cdagdn", s1) * op("Cdn", s2) +
         t * op("Cdn",    s1) * op("Cdagdn", s2) +
        Vpp * op("Nup", s1) * op("Nup", s2) +
        Vpp * op("Ndn", s1) * op("Ndn", s2) +
        Vpm * op("Nup", s1) * op("Ndn", s2) +
        Vpm * op("Ndn", s1) * op("Nup", s2)
    return hj
end

# ================================================================
# One-site Hubbard Hamiltonian
# ================================================================

function onsite_hamiltonian(sites, i, U)

    return U * op("Nupdn", sites[i])

end


function build_tebd_gates_2nd(sites, dt, t, U, Vpp, Vpm)
    N = length(sites)
    gates = ITensor[]

    for i in 1:N
        hi = onsite_hamiltonian(sites, i, U)
        Gi = exp(-im * dt / 2 * hi)
        push!(gates, Gi)
    end

    bond_gates = ITensor[]
    for i in 1:(N - 1)
        hi = bond_hamiltonian(sites,i,t,Vpp,Vpm)
        Gi = exp(-im * dt / 2 * hi)
        push!(bond_gates, Gi)
    end
    append!(gates, bond_gates)
    append!(gates, reverse(bond_gates))

    for i in 1:N
        hi = onsite_hamiltonian(sites, i, U)
        Gi = exp(-im * dt / 2 * hi)
        push!(gates, Gi)
    end
    return gates
end

# ================================================================
function build_tebd_gates_4th(sites, dt,t, U, Vpp, Vpm)
    p = 1 / (4 - 4^(1 / 3))
    q = 1 - 4p
    gates = ITensor[]
    for scale in (p, p, q, p, p)
        append!(
            gates,
            build_tebd_gates_2nd(
                sites,
                scale * dt,
                t,
                U,
                Vpp,
                Vpm,
            ),
        )
    end
    return gates
end


let
    N = 64
    t, U, Vpp, Vpm = 1.0, 0.1, 0.8, 4.0
    filename = filename_builder(N, t, U, Vpp, Vpm)
    psi0, params = load_simulation(filename, Val(:all))
    println("The initial bond dimension of psi0 is ", ret_maxlinkdim(psi0))
    sites = siteinds(psi0)
    tau = 1E-4
    ttotal = 2 * tau

    cutoff = 1e-14
    maxdim = 600
    gates = build_tebd_gates_2nd(sites, tau, t, U, Vpp, Vpm)

    psi=copy(psi0)
    @profview begin
    for time in 0.0:tau:ttotal
    # Measure observables here
    # ...
    psi = apply(gates, psi; cutoff=cutoff, maxdim=maxdim)
    normalize!(psi)
    time≈ttotal && break
    inp = abs(inner(psi0', psi))
    @show inp
    end
    end
    println("The final bond dimesnion is ", ret_maxlinkdim(psi))
end
