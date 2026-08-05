using MKL
using Spin1LongRangeOrder
using ITensors, ITensorMPS
using Spin1LongRangeOrder.Hamiltonians
using HDF5
using LinearAlgebra
using Printf
using Plots
gr()



using ITensors, ITensorMPS

"""
    op_correlation(psi::MPS, sites, opA::Tuple{String,String}, opAdag::Tuple{String,String};
                   i0::Int=1, js=(i0+1):(length(psi)-1),
                   normalize::Bool=true, connected::Bool=false)

Compute ⟨O(j) O†(i0)⟩ for a finite MPS, where
    O(n)    = op(opA[1],    n)   * op(opA[2],    n+1)
    O†(n)   = op(opAdag[1], n+1) * op(opAdag[2], n)

Default example: O(n) = S+_n S+_{n+1}, O†(n) = S-_{n+1} S-_n ->
    opA    = ("S+","S+")
    opAdag = ("S-","S-")

Built via `apply` on single-site operators only (exact — no cutoff/maxdim
needed, no truncation), so it's valid regardless of whether O(j)'s support
overlaps O†(i0)'s support.

Returns
-------
x    :: Vector{Int}         -- same as `js`, as a plain array
corr :: Vector{ComplexF64}  -- corr[k] = ⟨O(x[k]) O†(i0)⟩
"""
function op_correlation(psi::MPS, 
                         opA::Tuple{String,String}=("S+","S+"),
                         opAdag::Tuple{String,String}=("S-","S-");
                         i0::Int=1,
                         js=(i0+2):(length(psi)-1),
                         normalize::Bool=true,
                         connected::Bool=false)

    sites = siteinds(psi)
    Odag_gates(n) = [op(opAdag[2], sites[n+1]), op(opAdag[1], sites[n])]
    O_gates(n)    = [op(opA[2],    sites[n+1]), op(opA[1],    sites[n])]

    psi0  = apply(Odag_gates(i0), psi)          # O†(i0) |psi>
    norm2 = normalize ? inner(psi, psi) : one(ComplexF64)

    x    = collect(js)
    corr = Vector{ComplexF64}(undef, length(x))

    for (k, j) in enumerate(x)
        psij   = apply(Odag_gates(j), psi)       # O(j)† |psi> = O†(j)|psi> -- used as bra for O(j)
        corr[k] = inner(psij, psi0) / norm2
    end

    if connected
        # <O> and <O†> evaluated the same way, then subtract <O><O†>
        psiO_i0  = apply(O_gates(i0), psi)
        O_val    = inner(psi, psiO_i0) / norm2
        Odag_val = inner(psi, psi0)    / norm2
        corr .-= O_val * Odag_val
    end

    return x, corr
end



##
let
    dataroot = "data/Hubbard/"
    N = 128
    Npart = N ÷ 2
    sampledatafile = joinpath(dataroot, "N$(N)consNf/N$(N)_U0.100_Vpp0.800_Vpm4.350_Np$(Npart).h5")
    psi, params = load_simulation(sampledatafile, Val(:params))
    i0 = N ÷ 2
    js = (i0+2):(length(psi)-1)
    xvals, corr = op_correlation(psi, siteinds(psi), ("S+","S+"), ("S-","S-"), i0=i0, js=js)
    p = plot(1:length(corr), abs.(corr); lw=2, marker=:circle, label="|⟨S⁺S⁺⟩|", xlabel="x")
    plot!(p; yscale=:log, xscale=:log)
    display(p)
end

