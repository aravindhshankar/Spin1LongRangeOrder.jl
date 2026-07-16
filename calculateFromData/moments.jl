using MKL
using Spin1LongRangeOrder
using ITensors, ITensorMPS
using Spin1LongRangeOrder.Hamiltonians
using HDF5
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "correlations.jl"))  # reuse filename_builder, load_simulation, etc.

function moments_output_filename(datafilename)
    # e.g. data/Hubbard/N256consNf/N256_U0.100_Vpp0.800_Vpm4.000_Np128.h5
    #   -> data/Hubbard/moments/N256_U0.100_Vpp0.800_Vpm4.000_Np128_moments.h5
    _, fname = splitdir(datafilename)
    outdir = joinpath("data", "Hubbard", "moments")
    mkpath(outdir)
    return joinpath(outdir, replace(fname, ".h5" => "_moments.h5"))
end

"""
    compute_and_save_moments(datafilename; mpo_cutoff=1e-12, blas_threads_per_task=1)
"""
function compute_and_save_moments(datafilename; mpo_cutoff::Float64=0.0,
                                   blas_threads_per_task::Int=1)
    outfile = moments_output_filename(datafilename)

    if isfile(outfile)
        println("Already computed, skipping: $outfile")
        return outfile
    end

    psi, params = load_simulation(datafilename, Val(:all))
    println("Loaded simulation from $datafilename")

    BLAS.set_num_threads(blas_threads_per_task)

    N = length(psi)
    sites = siteinds(psi)

    # Build MPO for total magnetization M = sum_i Sz_i (bond dimension 2)
    os = OpSum()
    for i in 1:N
        os += 1.0, "Sz", i
    end
    Mmpo = MPO(os, sites)

    # Exact MPO*MPO products — these are tiny (bond <=4, <=16) regardless of N,
    # since they're built by operator multiplication, not by enumerating terms.
    try
        M2mpo = apply(Mmpo, Mmpo; cutoff=mpo_cutoff)   
    catch e 
        println("ERROR processing $datafilename: $e")
        println("Defaulting to naive algorithm for tensor contraction in M2mpo")
        M2mpo = apply(Mmpo, Mmpo; cutoff=0.0, alg="naive")
    end
    try 
        M4mpo = apply(M2mpo, M2mpo; cutoff=mpo_cutoff)
    catch e
        println("ERROR processing $datafilename: $e")
        println("Defaulting to naive algorithm for tensor contraction in M2mpo")
        M4mpo = apply(M2mpo, M2mpo; cutoff=0.0, alg="naive")
    end
    println("  built M^2 MPO (maxlinkdim=$(maxlinkdim(M2mpo))), " *
            "M^4 MPO (maxlinkdim=$(maxlinkdim(M4mpo)))")

    t0 = time()
    m2_raw = inner(psi', M2mpo, psi)
    m4_raw = inner(psi', M4mpo, psi)
    println(@sprintf("  <M^2>, <M^4> computed in %.1f s (chi_psi=%d)", time() - t0, maxlinkdim(psi)))

    m2 = m2_raw / N^2
    m4 = m4_raw / N^4

    h5open(outfile, "w") do f
        f["N"]  = N
        f["m2"] = m2
        f["m4"] = m4
        f["m2_raw"] = m2_raw   # <M^2>, unnormalized, kept for debugging
        f["m4_raw"] = m4_raw   # <M^4>, unnormalized, kept for debugging
        f["mpo_cutoff"] = mpo_cutoff
        for (k, v) in params
            try
                f["params/$k"] = v
            catch
                f["params/$k"] = string(v)
            end
        end
    end

    println("Saved moments to $outfile  (m2=$m2, m4=$m4)")
    return outfile
end

function load_moments(datafilename)
    momfile = moments_output_filename(datafilename)
    if !isfile(momfile)
        error("No moments file found at $momfile — did you compute/download it?")
    end

    data = h5open(momfile, "r") do f
        (
            N  = read(f["N"]),
            m2 = read(f["m2"]),
            m4 = read(f["m4"]),
        )
    end
    return data
end
