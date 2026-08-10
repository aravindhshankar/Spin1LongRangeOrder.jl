using MKL
using Spin1LongRangeOrder
using ITensors, ITensorMPS
using Spin1LongRangeOrder.Hamiltonians
using HDF5
using LinearAlgebra
using Printf

fmt(x) = replace(@sprintf(" % .3f", x), " " => "")
function filename_builder(N, g1, g2)
    filename = joinpath("data", "N$(N)", "g1_$(fmt(g1))_g2_$(fmt(g2)).h5")
    return filename
end

function corr_output_filename(datafilename)
    _, fname = splitdir(datafilename)
    outdir = joinpath("data", "Spin1corrs")
    mkpath(outdir)
    return joinpath(outdir, replace(fname, ".h5" => "_corrs.h5"))
end


function compute_and_save_correlations(datafilename; blas_threads_per_task::Int=1)
    outfile = corr_output_filename(datafilename)

    if isfile(outfile)
        println("Already computed, skipping: $outfile")
        return outfile
    end

    psi, params = load_simulation(datafilename, Val(:params))
    println("Loaded simulation from $datafilename")

    BLAS.set_num_threads(blas_threads_per_task)

    t_sz      = Threads.@spawn correlation_matrix(psi, "Sz", "Sz")
    t_sz2     = Threads.@spawn correlation_matrix(psi, "Sz2", "Sz2")
    t_spm     = Threads.@spawn correlation_matrix(psi, "S+", "S-")
    t_sx      = Threads.@spawn correlation_matrix(psi, "Sx", "Sx")
    t_sy      = Threads.@spawn correlation_matrix(psi, "Sy", "Sy")

    # cheap, do directly on main thread while the above run
    sz_exp   = expect(psi, "Sz")

    szmat     = fetch(t_sz)
    sz2mat    = fetch(t_sz2)
    spmmat    = fetch(t_spm)
    sxmat     = fetch(t_sx)
    symat     = fetch(t_sy)

    h5open(outfile, "w") do f
        f["szmat"]       = szmat
        f["sz2mat"]      = sz2mat
        f["spmmat"]      = spmmat
        f["sxmat"]       = sxmat
        f["symat"]       = symat
        f["sz_expect"]   = sz_exp
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
            sz2mat     = read(f["sz2mat"]),
            spmmat     = read(f["spmmat"]),
            sxmat      = read(f["sxmat"]),
            symat      = read(f["symat"]),
            sz_expect   = read(f["sz_expect"]),
        )
    end
    return data
end