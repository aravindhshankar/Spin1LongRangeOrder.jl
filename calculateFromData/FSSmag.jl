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

using LsqFit


using Optim
using Interpolations

function fit_fss(Ndict, dVlist; p0=nothing)

    # ------------------------------------------------------------
    # Flatten the data
    # ------------------------------------------------------------
    Ns = Float64[]
    dVs = Float64[]
    ms = Float64[]

    for (N, m) in Ndict
        @assert length(m) == length(dVlist)

        append!(Ns, fill(Float64(N), length(dVlist)))
        append!(dVs, dVlist)
        append!(ms, m)
    end

    # ------------------------------------------------------------
    # Given (dVstar, beta, nu), calculate the collapsed coordinates
    # ------------------------------------------------------------
    function collapsed(p)
        dVstar, beta, nu = p

        x = (dVs .- dVstar) .* Ns .^ (1 / nu)
        y = ms .* Ns .^ (beta / nu)

        return x, y
    end

    # ------------------------------------------------------------
    # Objective:
    #
    # For every system size, interpolate its collapsed curve and
    # compare it with the other system sizes.
    #
    # We use the largest-N curve as the reference.
    # ------------------------------------------------------------
    function objective(p)

        dVstar, beta, nu = p

        # Reject unphysical parameters
        if nu <= 0 || beta < 0
            return Inf
        end

        x, y = collapsed(p)

        # Split collapsed data by N
        curves = Dict{Float64,Tuple{Vector{Float64},Vector{Float64}}}()

        for N in keys(Ndict)
            mask = Ns .== Float64(N)

            xi = x[mask]
            yi = y[mask]

            # Sort by x
            perm = sortperm(xi)

            curves[Float64(N)] = (xi[perm], yi[perm])
        end

        # --------------------------------------------------------
        # Compare every pair of curves.
        # For each point on curve i, interpolate curve j.
        # --------------------------------------------------------
        err = 0.0
        npoints = 0

        sizes = collect(keys(curves))

        for i in 1:length(sizes)
            xi, yi = curves[Float64(sizes[i])]

            for j in (i+1):length(sizes)
                xj, yj = curves[Float64(sizes[j])]

                # Only compare over their common x range
                xmin = max(minimum(xi), minimum(xj))
                xmax = min(maximum(xi), maximum(xj))

                if xmax <= xmin
                    continue
                end

                mask = (xi .>= xmin) .& (xi .<= xmax)

                if !any(mask)
                    continue
                end

                # Linear interpolation of curve j
                interp = LinearInterpolation(xj, yj)

                residuals = yi[mask] .- interp.(xi[mask])

                err += sum(abs2, residuals)
                npoints += length(residuals)
            end
        end

        return npoints == 0 ? Inf : err / npoints
    end

    # ------------------------------------------------------------
    # Initial guess
    # ------------------------------------------------------------
    if p0 === nothing
        p0 = [
            3.55,   # dVstar
            1.0/3.0,            # beta
            7.0/12.0             # nu
        ]
    end

    # ------------------------------------------------------------
    # Optimize
    # ------------------------------------------------------------
    result = optimize(
        objective,
        p0,
        NelderMead()
    )

    p = Optim.minimizer(result)

    return (
        dVstar = p[1],
        beta   = p[2],
        nu     = p[3],
        error  = Optim.minimum(result),
        result = result
    )
end




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

    for N in Nlist
        magzlist = Float64[]
        for dV in dVlist
            datafilename = filename_builder(N, t, U, Vpp, dV)
            magz = 
            try 
                abs(sum(load_correlations(datafilename).sz_expect)/N)
            catch _ 
                nothing
            end
            push!(magzlist, magz)
        end
        # plot!(dVlist, magzlist, label="N=$N", markers=:circle)
        Ndict[N] = magzlist
    end

    beta = 1.0/3.0 - 0.017
    nu = 7.0/12.0 - 0.01
    gstar = 3.554
    fit = fit_fss(Ndict, dVlist; p0 = [gstar, beta, nu])
    gstar =  fit.dVstar
    beta = fit.beta
    nu = fit.nu

    println("dVstar = ", fit.dVstar)
    println("beta   = ", fit.beta)
    println("nu     = ", fit.nu)
    println("error  = ", fit.error)

    for N in Nlist 
        magzlist = Ndict[N]
        xvals = @. (dVlist - gstar) * ((N * 1.0) ^ (1.0/nu))
        yvals = @. (magzlist) * (N^(beta/nu)) 
        scatter!(xvals, yvals, label="N=$N", markers=:circle)
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
    # plot!(xlims=(-2500,2500))
    display(p)
    # @show Ndict[64]
end