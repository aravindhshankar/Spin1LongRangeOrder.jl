using MKL
using Spin1LongRangeOrder
using ITensors, ITensorMPS
using Spin1LongRangeOrder.Hamiltonians
using HDF5
using LinearAlgebra
using Printf

function filename_builder(N, t, U, Vpp, dV; prefix="data/Hubbard/", makepath=false)
    Vpm = Vpp + dV
    Npart = Int(N // 2)
    _ = t # not used, so we discard, but leave the API as is for the future
    datasavedir = joinpath(prefix, "N$N"*"consNf/")
    makepath && mkpath(datasavedir)
    datafilename = datasavedir * "N$N" * "_U" * @sprintf("%.3f", U) * "_Vpp" * @sprintf("%.3f", Vpp) * "_Vpm" * @sprintf("%.3f", Vpm) * "_Np$Npart" * raw".h5"
    return datafilename
end

function corr_output_filename(datafilename)
    # e.g. data/Hubbard/N256consNf/N256_U0.100_Vpp0.800_Vpm4.000_Np128.h5
    #   -> data/Hubbard/corrs/N256_U0.100_Vpp0.800_Vpm4.000_Np128_corrs.h5
    dirtree, fname = splitdir(datafilename)
    if occursin("ImpPrec", dirtree)
        outdir = joinpath("data", "Hubbard", "corrs_imp")
    else
        outdir = joinpath("data", "Hubbard", "corrs")
    end
    mkpath(outdir)
    return joinpath(outdir, replace(fname, ".h5" => "_corrs.h5"))
end


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


function compute_and_save_correlations(datafilename; blas_threads_per_task::Int=1, skipout::Bool=true)
    outfile = corr_output_filename(datafilename)

    if skipout && isfile(outfile)
        println("Already computed, skipping: $outfile")
        return outfile
    end

    psi, params = load_simulation(datafilename, Val(:all))
    println("Loaded simulation from $datafilename")

    BLAS.set_num_threads(blas_threads_per_task)

    N = length(psi)
    i0 = N ÷ 2
    js = (i0+2):(N-1)
    # Full correlation matrices (N x N each) — no site selection here anymore
    t_sz      = Threads.@spawn correlation_matrix(psi, "Sz", "Sz")
    t_spm     = Threads.@spawn correlation_matrix(psi, "S+", "S-")
    t_charge  = Threads.@spawn correlation_matrix(psi, "Ntot", "Ntot")
    t_elec_up = Threads.@spawn correlation_matrix(psi, "Cdagup", "Cup")
    t_elec_dn = Threads.@spawn correlation_matrix(psi, "Cdagdn", "Cdn")
    t_spin_2  = Threads.@spawn op_correlation(psi, ("S+","S+"), ("S-","S-"), i0=i0, js=js)   

    # cheap, do directly on main thread while the above run
    sz_exp   = expect(psi, "Sz")
    ntot_exp = expect(psi, "Ntot")

    szmat      = fetch(t_sz)
    spmmat     = fetch(t_spm)
    chargemat  = fetch(t_charge)
    elecupmat  = fetch(t_elec_up)
    elecdnmat  = fetch(t_elec_dn)
    xspin2, spin2vec   = fetch(t_spin_2)
    h5open(outfile, "w") do f
        f["szmat"]       = szmat
        f["chargemat"]   = chargemat
        f["spmmat"]      = spmmat
        f["elecupmat"]   = elecupmat
        f["elecdnmat"]   = elecdnmat
        f["xspin2"]       = xspin2
        f["spin2vec"]     = spin2vec
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
            szmat       = read(f["szmat"]),
            chargemat   = read(f["chargemat"]),
            spmmat      = read(f["spmmat"]),
            elecupmat   = read(f["elecupmat"]),
            elecdnmat   = read(f["elecdnmat"]),
            sz_expect   = read(f["sz_expect"]),
            ntot_expect = read(f["ntot_expect"]),
            xspin2      = haskey(f, "xspin2") ? read(f["xspin2"]) : missing,
            spin2vec    = haskey(f, "spin2vec") ? read(f["spin2vec"]) : missing,
        )
    end
    return data
end