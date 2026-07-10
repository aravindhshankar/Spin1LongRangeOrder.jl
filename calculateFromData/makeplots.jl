using HDF5
using Plots
using Printf
gr()

include(joinpath(@__DIR__, "correlations.jl"))  # for corr_output_filename, filename_builder

function load_correlations(datafilename)
    corrfile = corr_output_filename(datafilename)
    if !isfile(corrfile)
        error("No correlation file found at $corrfile")
    end

    data = h5open(corrfile, "r") do f
        (
            szcorr_full     = read(f["szcorr_full"]),
            szcorr_conn     = read(f["szcorr_conn"]),
            chargecorr_full = read(f["chargecorr_full"]),
            chargecorr_conn = read(f["chargecorr_conn"]),
            spmcorr          = read(f["spmcorr"]),
            electroncorr_up  = read(f["electroncorr_up"]),
            electroncorr_dn  = read(f["electroncorr_dn"]),
            site              = read(f["site"]),
        )
    end
    return data
end

function plot_correlations(datafilename; title_prefix="")
    data = load_correlations(datafilename)
    site = data.site
    N = length(data.szcorr_full)
    xvals = 1:N

    # distance from reference site, used as the "site" axis for log-log plots
    dist = abs.(xvals .- site)
    # avoid log(0) at the reference site itself
    keep = dist .> 0

    panels = [
        ("Sz-Sz (full)",        data.szcorr_full),
        ("Sz-Sz (connected)",   data.szcorr_conn),
        ("Ntot-Ntot (full)",    data.chargecorr_full),
        ("Ntot-Ntot (connected)", data.chargecorr_conn),
        ("S+ - S-",             data.spmcorr),
        ("Cdagup-Cup",          data.electroncorr_up),
        ("Cdagdn-Cdn",          data.electroncorr_dn),
    ]

    plts = Plots.Plot[]
    for (label, y) in panels
        yv = abs.(y[keep])
        xv = dist[keep]
        p = plot(xv, yv,
            xscale=:log10, yscale=:log10,
            xlabel="|site - $site|", ylabel="|correlation|",
            title=label, legend=false, lw=2, ms=4, marker=:circle)
        push!(plts, p)
    end

    fullplot = plot(plts..., layout=(4, 2), size=(1000, 1400),
        plot_title="$title_prefix (ref site $site)")
    display(fullplot)
    return fullplot
end

let 
    N = 256
    U = 0.1
    Vpp = 0.8
    dV = 3.2
    t = 1.0
    datafilename = filename_builder(N, t, U, Vpp, dV)

    plot_correlations(datafilename; title_prefix="N=$N, U=$U, Vpp=$Vpp, dV=$dV")
end