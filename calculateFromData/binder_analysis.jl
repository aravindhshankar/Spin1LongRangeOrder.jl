using HDF5
using Plots
using Printf
using LaTeXStrings
using Statistics
gr()

include(joinpath(@__DIR__, "moments.jl"))  # for filename_builder, load_moments (cheap, local)

binder_cumulant(m2, m4) = 1 - m4 / (3 * m2^2)

"""
    crossing(dVlist, yA, yB)

Finds dV where curves yA(dV) and yB(dV) cross, by linear interpolation between
grid points where the sign of (yA - yB) flips. Returns a vector (there may be
more than one crossing, or none, in the given range).
"""
function crossing(dVlist, yA, yB)
    diffs = yA .- yB
    crossings = Float64[]
    for i in 1:length(dVlist)-1
        d0, d1 = diffs[i], diffs[i+1]
        if d0 == 0
            push!(crossings, dVlist[i])
        elseif sign(d0) != sign(d1)
            x0, x1 = dVlist[i], dVlist[i+1]
            xc = x0 - d0 * (x1 - x0) / (d1 - d0)
            push!(crossings, xc)
        end
    end
    return crossings
end

##
let
    U = 0.1
    Vpp = 0.8
    t = 1.0
    dVlist = collect(3.2:0.05:3.9)          # must match the scan used in driver_moments.jl
    Nlist = (64, 128, 256)

    binder = Dict{Int,Vector{Float64}}()
    for N in Nlist
        blist = Float64[]
        for dV in dVlist
            datafilename = filename_builder(N, t, U, Vpp, dV)
            try 
                data = load_moments(datafilename) 
                push!(blist, binder_cumulant(data.m2, data.m4))    # cheap: just reads two scalars per file
            catch e 
                println(e)
            end
                
        end
        binder[N] = blist
    end

    p = plot()
    for N in Nlist
        plot!(dVlist, binder[N], label="N=$N", marker=:circle, markersize=3)
    end
    plot!(xlabel=L"\delta V", ylabel=L"U_L = 1 - \langle m^4\rangle / 3\langle m^2\rangle^2",
          title="Binder cumulant crossings", legend=:best)
    plot!(ylims=(0,1.1))
    println("Pairwise crossings between adjacent sizes:")
    Nsorted = sort(collect(Nlist))
    all_crossings = Float64[]
    for k in 1:length(Nsorted)-1
        Na, Nb = Nsorted[k], Nsorted[k+1]
        xcs = crossing(dVlist, binder[Na], binder[Nb])
        if isempty(xcs)
            println("  N=$Na vs N=$Nb: no crossing found in [$(dVlist[1]), $(dVlist[end])]")
        end
        for xc in xcs
            @printf("  N=%d vs N=%d: dVc = %.4f\n", Na, Nb, xc)
            push!(all_crossings, xc)
        end
    end

    if !isempty(all_crossings)
        gstar_est = mean(all_crossings)
        gstar_err = length(all_crossings) > 1 ? std(all_crossings) / sqrt(length(all_crossings)) : 0.0
        @printf("\nEstimated critical point (mean of pairwise crossings): dVc = %.4f +/- %.4f\n",
                gstar_est, gstar_err)
        vline!([gstar_est], ls=:dash, lc=:black, label=@sprintf("dVc = %.4f", gstar_est))
    else
        println("\nNo crossings found anywhere — widen dVlist in both this file and driver_moments.jl.")
    end

    display(p)
end
