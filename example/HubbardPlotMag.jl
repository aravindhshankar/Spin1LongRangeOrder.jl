using MKL
using Spin1LongRangeOrder
using ITensors, ITensorMPS
using HDF5
using Plots
gr()
## 
let 
    Vpp=0.8
    U=0.1
    N=64
    Nf=32
    datadir = "data/Hubbard/N64consNf/"
    h5files = filter(f -> endswith(f, ".h5"),
                 readdir(datadir; join=true))

    dVs = Float64[]
    Szs = Float64[]
    Es  = Float64[]
    _, parmforkeys = load_simulation(h5files[1], Val(:params))
    println(collect(keys(parmforkeys)))

    for file in h5files
        try
            _, params = load_simulation(file, Val(:params))

            push!(dVs, params["dV"])
            push!(Szs, params["totalSz"])
            push!(Es, params["E"])
            println("dV = ", params["dV"], ", var= ", params["var"])
        catch err
            @warn "Skipping $file" exception=err
        end
    end
    perm = sortperm(dVs)
    dVs = dVs[perm]
    Szs = Szs[perm]
    p = plot(dVs, Szs, 
            marker=:circle,
            markersize=4,
            linewidth=2,
            xlabel="dV",
            ylabel="totalSz",
            # ylabel = "Energy",
            legend=false, 
            color=:orange
        )
    hline!([0.25],
       linestyle=:dot,
       color=:gray,
       linewidth=2,
       label="")
    title!("N=$N" * ", U=$U" * ", Vpp=$Vpp" * ", Nf=$Nf")
    savefig("pngfigs/N64Np32.png")
    # savefig("pngfigs/EnsN64Np32.png")
    display(p)
end
