using HDF5
using Plots
using Printf
using Polynomials
gr()

include(joinpath(@__DIR__, "correlations.jl"))  # for corr_output_filename, filename_builder

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

function plot_correlations(datafilename; site::Int, title_prefix="", nfit::Int=100)
    data = load_correlations(datafilename)
    N = size(data.szmat, 1)
    xvals = 1:N

    # connected correlators built here, now that site is chosen
    szcorr_conn     = data.szmat[site, :]     .- data.sz_expect[site]   .* data.sz_expect
    chargecorr_conn = data.chargemat[site, :] .- data.ntot_expect[site] .* data.ntot_expect

    dist = xvals .- site
    keep = dist .> 0

    panels = [
        ("Sz-Sz (full)",          data.szmat[site, :]),
        ("Sz-Sz (connected)",     szcorr_conn),
        ("Ntot-Ntot (full)",      data.chargemat[site, :]),
        ("Ntot-Ntot (connected)", chargecorr_conn),
        ("S+ - S-",               data.spmmat[site, :]),
        ("Cdagup-Cup",            data.elecupmat[site, :]),
        ("Cdagdn-Cdn",            data.elecdnmat[site, :]),
    ]

    plts = Plots.Plot[]
    for (label, y) in panels
        xv = dist[keep]
        yv = abs.(y[keep])

        posmask = yv .> 0
        xv_pos, yv_pos = xv[posmask], yv[posmask]

        n_use = min(nfit, length(xv_pos))
        p = plot(xv, yv,
            xscale=:log10, yscale=:log10,
            xlabel="site - $site", ylabel="|correlation|",
            title=label, lw=2, ms=4, marker=:circle,
            label="data", legendfontsize=10)

        if n_use >= 2
            xfit = xv_pos[1:n_use]
            yfit = yv_pos[1:n_use]
            slope, intercept = loglog_fit(xfit, yfit)
            yline = 10 .^ (slope .* log10.(xfit) .+ intercept)
            fitlabel = @sprintf("slope=%.2f", slope)
            plot!(p, xfit, yline, ls=:dash, lw=2, color=:red, label=fitlabel)
        end

        push!(plts, p)
    end

    fullplot = plot(plts..., layout=(4, 2), size=(1100, 1500),
        plot_title="$title_prefix (ref site $site)")
    display(fullplot)
    return fullplot
end

##
let 
    N = 256
    U = 0.1
    Vpp = 0.8
    dV = 3.55 #choices 3.2, 3.5, 3.55, 3.7
    t = 1.0
    datafilename = filename_builder(N, t, U, Vpp, dV)

    plot_correlations(datafilename; title_prefix="N=$N, U=$U, Vpp=$Vpp, dV=$dV", nfit=50)
end