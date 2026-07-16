using HDF5
using Plots
using Polynomials
using Printf
using LaTeXStrings
using FFTW
gr()
# pyplot()

include(joinpath(@__DIR__, "correlations.jl"))  # for corr_output_filename, filename_builder, load_correlations


# let 
#     U = 0.1
#     Vpp = 0.8
#     t = 1.0
#     dVlist = 3.2:0.05:3.9
#     # dVlist = (3.4, 3.5, 3.55, 3.6, 3.65, 3.85)
#     N = 256
#     magzlist = Float64[]
#     for dV in dVlist
#         datafilename = filename_builder(N, t, U, Vpp, dV)
#         magz = abs.(sum(load_correlations(datafilename).sz_expect) / N)
#         push!(magzlist, magz)
#     end
#     plot(dVlist, magzlist)
#     dVcrit = 3.545
#     beta = 1.0/3
#     fsscurve = @. 0.35 * abs(1 - dVlist/dVcrit)^(beta)
#     plot!(dVlist, fsscurve, ls=:dash)
# end

##
let 
    U = 0.1
    Vpp = 0.8
    t = 1.0
    dVlist = 3.2:0.05:3.9
    # dVlist = (3.4, 3.5, 3.55, 3.6, 3.65, 3.85)
    Nlist = (64, 128, 256)
    # p = plot()
    p = scatter()
    Ndict = Dict()
    beta = 1.0/3.0 + 0.1
    nu = 7.0/12 + 0.02
    gstar = 3.535
    for N in Nlist
        magzlist = Float64[]
        for dV in dVlist
            datafilename = filename_builder(N, t, U, Vpp, dV)
            magz = abs(sum(load_correlations(datafilename).sz_expect)/N)
            push!(magzlist, magz)
        end
        # plot!(dVlist, magzlist, label="N=$N", markers=:circle)
        xvals = @. (dVlist - gstar) * (N ^ (1.0/nu))
        yvals = @. (magzlist) * (N^(beta/nu)) 
        scatter!(xvals, yvals, label="N=$N", markers=:circle)
        Ndict[N] = magzlist
    end
    scatter!(xlabel=L"(g-g_c)L^{1/\nu}")
    scatter!(ylabel=L"L^{\beta/\nu}m(g,L)")
    scatter!(title=L"g_c="*@sprintf("%.4f", gstar))
    ltest=1000
    densedVlist = range(3.51,3.64,length=1000)
    xvals = @. (densedVlist - gstar) * (ltest ^ (1.0/nu))
    filter = densedVlist .> gstar
    yvals = @. 0.270*((densedVlist[densedVlist.>gstar] - gstar) ^ beta) * (ltest ^ (beta/nu))
    plot!(xvals[filter], yvals)
    plot!(xlims=(-2500,2500))
    display(p)
    # @show Ndict
end