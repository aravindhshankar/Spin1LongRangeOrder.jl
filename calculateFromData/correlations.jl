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


function compute_and_save_correlations(datafilename; blas_threads_per_task::Int=1)
    outfile = corr_output_filename(datafilename)

    if isfile(outfile)
        println("Already computed, skipping: $outfile")
        return outfile
    end

    psi, params = load_simulation(datafilename, Val(:all))
    println("Loaded simulation from $datafilename")

    BLAS.set_num_threads(blas_threads_per_task)

    # Full correlation matrices (N x N each) — no site selection here anymore
    t_sz      = Threads.@spawn correlation_matrix(psi, "Sz", "Sz")
    t_spm     = Threads.@spawn correlation_matrix(psi, "S+", "S-")
    t_charge  = Threads.@spawn correlation_matrix(psi, "Ntot", "Ntot")
    t_elec_up = Threads.@spawn correlation_matrix(psi, "Cdagup", "Cup")
    t_elec_dn = Threads.@spawn correlation_matrix(psi, "Cdagdn", "Cdn")

    # cheap, do directly on main thread while the above run
    sz_exp   = expect(psi, "Sz")
    ntot_exp = expect(psi, "Ntot")

    szmat      = fetch(t_sz)
    spmmat     = fetch(t_spm)
    chargemat  = fetch(t_charge)
    elecupmat  = fetch(t_elec_up)
    elecdnmat  = fetch(t_elec_dn)

    h5open(outfile, "w") do f
        f["szmat"]       = szmat
        f["chargemat"]   = chargemat
        f["spmmat"]      = spmmat
        f["elecupmat"]   = elecupmat
        f["elecdnmat"]   = elecdnmat
        f["sz_expect"]   = sz_exp
        f["ntot_expect"] = ntot_exp
        for (k, v) in params
            try
                f["params/$k"] = v
            catch
                f["params/$k"] = string(v)
            end
        end
    end

    println("Saved correlation matrices to $outfile")
    return outfile
end

function loglog_fit(xv, yv)
    lx = log10.(xv)
    ly = log10.(yv)
    A = hcat(lx, ones(length(lx)))  # design matrix [log10(x)  1]
    coeffs = A \ ly                  # least squares solve
    slope, intercept = coeffs[1], coeffs[2]
    return slope, intercept
end


function load_correlations(datafilename)
    corrfile = corr_output_filename(datafilename)
    if !isfile(corrfile)
        error("No correlation file found at $corrfile — did you download it?")
    end

    data = h5open(corrfile, "r") do f
        (
            szmat      = read(f["szmat"]),
            chargemat  = read(f["chargemat"]),
            spmmat     = read(f["spmmat"]),
            elecupmat  = read(f["elecupmat"]),
            elecdnmat  = read(f["elecdnmat"]),
            sz_expect   = read(f["sz_expect"]),
            ntot_expect = read(f["ntot_expect"]),
        )
    end
    return data
end