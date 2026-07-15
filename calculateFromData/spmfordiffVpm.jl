using HDF5
using Plots
using Polynomials
using Printf
using LaTeXStrings
using FFTW
gr()
# pyplot()

include(joinpath(@__DIR__, "correlations.jl"))  # for corr_output_filename, filename_builder, load_correlations
##
let 
    #### PARA CORRELATION FUNCTIONS ############
    p = plot()
    # colordict = Dict(64=>:blue, 128=>:green, 256=>:red)
    maxdist = range(1, stop=128, length=128)
    N = 256
    site = Int(N//2)
    xvals = 1:N
        U = 0.1
    Vpp = 0.8
    t = 1.0
    # dVlist = 3.2:0.1:3.9
    dVlist = (3.4, 3.5, 3.55, 3.6, 3.65, 3.85)
    for dV in dVlist

        datafilename = filename_builder(N, t, U, Vpp, dV)

        data = load_correlations(datafilename)
        # chargecorr_conn = data.chargemat[site, :] .- data.ntot_expect[site] .* data.ntot_expect
        spmcorr = data.spmmat[site, :]
        chargecorr_conn = spmcorr
        @show length(chargecorr_conn)
        dist = xvals .- site
        keep = dist .> 0
        keep_neg = keep #.& (chargecorr_conn .< 0)
        @show length(chargecorr_conn[keep_neg])
        plot!(dist[keep_neg][1:2:end], abs.(-chargecorr_conn[keep_neg])[1:2:end], 
            # marker=:circle, 
            label=@sprintf("dV=%.2f", dV),
        #color=colordict[N], strokecolor=colordict[N], lw=2, ms=2, label="N=$N"
        )
    end
    plot!(yscale=:log10, xscale=:log10)
    # plot!(yscale=:linear, xscale=:linear)
    # plot!(xlabel=raw"$x-\frac{N}{2}$", ylabel=raw"$| \delta\rho(x)\delta\rho(N/2)|$", title="Connected charge correlations N=$N", legendfontsize=8)
    plot!(xlabel=raw"$x-\frac{N}{2}$", ylabel=raw"$| S^+(x)S^-(N/2)|$", title="Spin +- correlations", legendfontsize=12)
    plot!(legend=:outerbottom, legendcolumns=6)
    # fitline = @. -0.21/ (pi * maxdist^2) + 0.01  * cos(0.0 * pi * maxdist) * maxdist^(-1 - 0.21*pi) / (log(maxdist))^1.5
    # fitline = @. 0.07 * maxdist^(-2)
    # plot!(maxdist, abs.(fitline), ls=:dash, lw=2, color=:black, label=raw"$\sim x^{-2}$")
    # plot!(maxdist, 0.0034 * maxdist.^(-1.21), ls=:dash, lw=2, color=:black, label=raw"$\sim x^{1.21}$")
end