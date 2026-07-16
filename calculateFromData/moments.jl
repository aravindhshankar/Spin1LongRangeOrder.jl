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

Computes <M^2> and <M^4> where M = sum_i Sz_i is the total magnetization operator.

IMPORTANT: this does NOT apply the M-MPO to psi (that would grow psi's bond dimension
by up to 2x per application — up to 4x for M^2 — forcing a lossy truncation of an
already-large state). Instead it computes the expectation values as an EXACT
bra-MPO-ket sandwich, <psi|M^n|psi> = inner(psi, Mn_mpo, psi), which never touches
psi's bond dimension at all. The only thing that needs to stay small is the MPO
itself, and it does:
  - M      has bond dimension 2  (sum of single-site terms)
  - M^2 = M*M   is an exact MPO product, bond dimension <= 4
  - M^4 = M^2*M^2 is an exact MPO product, bond dimension <= 16
These are literal operator multiplications (M*M as full quantum operators), not term
enumeration, so they correctly include all N^2 / N^4 cross terms (including i=j etc.)
without ever building an N^2- or N^4-term OpSum. Cost of the final inner() sandwiches
scales like N * chi_psi^2 * chi_MPO^2 — exact, no truncation error, no maxdim to tune.

"""
function compute_and_save_moments(datafilename; mpo_cutoff::Float64=1e-12,
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
    M2mpo = apply(Mmpo, Mmpo; cutoff=mpo_cutoff)   # MPO-MPO multiply, bond <= 4
    M4mpo = apply(M2mpo, M2mpo; cutoff=mpo_cutoff) # MPO-MPO multiply, bond <= 16
    println("  built M^2 MPO (maxlinkdim=$(maxlinkdim(M2mpo))), " *
            "M^4 MPO (maxlinkdim=$(maxlinkdim(M4mpo)))")

    # Exact sandwich contractions — psi's bond dimension is never modified/truncated.
    t0 = time()
    m2_raw = inner(psi, M2mpo, psi)
    m4_raw = inner(psi, M4mpo, psi)
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
