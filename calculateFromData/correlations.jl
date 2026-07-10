using MKL
using Spin1LongRangeOrder
using ITensors, ITensorMPS
using Spin1LongRangeOrder.Hamiltonians
using HDF5
using LinearAlgebra
using Printf

function filename_builder(N, t, U, Vpp, dV)
    Vpm = Vpp + dV
    Npart = Int(N // 2)
    _ = t
    datasavedir = "data/Hubbard/N$N" * "consNf/"
    datafilename = datasavedir * "N$N" * "_U" * @sprintf("%.3f", U) * "_Vpp" * @sprintf("%.3f", Vpp) * "_Vpm" * @sprintf("%.3f", Vpm) * "_Np$Npart" * raw".h5"
    return datafilename
end

function corr_output_filename(datafilename)
    # e.g. data/Hubbard/N256consNf/N256_U0.100_Vpp0.800_Vpm4.000_Np128.h5
    #   -> data/Hubbard/corrs/N256_U0.100_Vpp0.800_Vpm4.000_Np128_corrs.h5
    _, fname = splitdir(datafilename)
    outdir = joinpath("data", "Hubbard", "corrs")
    mkpath(outdir)
    return joinpath(outdir, replace(fname, ".h5" => "_corrs.h5"))
end

function compute_and_save_correlations(datafilename; blas_threads_per_task::Int=1, site::Int=20)
    outfile = corr_output_filename(datafilename)

    if isfile(outfile)
        println("Already computed, skipping: $outfile")
        return outfile
    end

    psi, params = load_simulation(datafilename, Val(:all))
    println("Loaded simulation from $datafilename")

    BLAS.set_num_threads(blas_threads_per_task)

    # Pairwise correlation matrices
    t_sz       = Threads.@spawn correlation_matrix(psi, "Sz", "Sz")[site, :]
    t_spm      = Threads.@spawn correlation_matrix(psi, "S+", "S-")[site, :]
    t_charge   = Threads.@spawn correlation_matrix(psi, "Ntot", "Ntot")[site, :]
    t_elec_up  = Threads.@spawn correlation_matrix(psi, "Cdagup", "Cup")[site, :]
    t_elec_dn  = Threads.@spawn correlation_matrix(psi, "Cdagdn", "Cdn")[site, :]

    # Single-site expectation values for connected correlators
    sz_exp   = expect(psi, "Sz")
    ntot_exp = expect(psi, "Ntot")

    szcorr_full     = fetch(t_sz)
    spmcorr         = fetch(t_spm)
    chargecorr_full = fetch(t_charge)
    electroncorr_up = fetch(t_elec_up)
    electroncorr_dn = fetch(t_elec_dn)


    # Connected correlators
    szcorr_conn     = szcorr_full     .- sz_exp[site]   .* sz_exp
    chargecorr_conn = chargecorr_full .- ntot_exp[site] .* ntot_exp

    h5open(outfile, "w") do f
        f["szcorr_full"]       = szcorr_full
        f["szcorr_conn"]       = szcorr_conn
        f["chargecorr_full"]   = chargecorr_full
        f["chargecorr_conn"]   = chargecorr_conn
        f["spmcorr"]           = spmcorr
        f["electroncorr_up"]   = electroncorr_up
        f["electroncorr_dn"]   = electroncorr_dn
        f["sz_expect"]         = sz_exp
        f["ntot_expect"]       = ntot_exp
        f["site"]              = site
        for (k, v) in params
            try
                f["params/$k"] = v
            catch
                f["params/$k"] = string(v)
            end
        end
    end

    println("Saved correlations to $outfile")
    return outfile
end