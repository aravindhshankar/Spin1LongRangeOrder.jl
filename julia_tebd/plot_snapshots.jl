using HDF5, Plots
gr()

function load_snapshots(path)
    steps = parse.(Int, [split(k, "_")[2] for k in keys(h5open(path, "r"))])
    order = sortperm(steps)
    ts, Cxs = Float64[], Vector{ComplexF64}[]
    h5open(path, "r") do f
        for k in keys(f)[order]
            g = f[k]
            push!(ts, read(g, "t"))
            push!(Cxs, complex.(read(g, "Cx_real"), read(g, "Cx_imag")))
        end
    end
    order2 = sortperm(ts)
    return ts[order2], Cxs[order2]
end

function plot_dynamical_quantities(DATAROOT, tag)
    operators = ["charge", "cdn", "Splus", "Sz"]
    plots = Plots.Plot[]
    for op in operators
        path = joinpath(DATAROOT, "dyn_corr", "$(tag)__op=$(op).h5")
        if !isfile(path)
            println("no data for $op yet")
            continue
        end
        ts, Cxs = load_snapshots(path)
        p = plot(title=op, xlabel="x", ylabel="|C(x,t)|", yscale=:log10, legend=:outertopright)
        for (tval, Cx) in zip(ts, Cxs)
            plot!(p, abs.(Cx), label="t=$(round(tval,digits=2))")
        end
        push!(plots, p)
    end
    plot(plots...; layout=(2, 2), size=(1100, 850))
end

# Example:
# using Spin1LongRangeOrder
# include(joinpath("../calculateFromData", "correlations.jl"))
# tag = basename(filename_builder(64, 1.0, 0.1, 0.8, 4.0))
# display(plot_dynamical_quantities("/path/to/dataroot", tag))
