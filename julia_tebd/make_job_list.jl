include("params_grid.jl")

outpath = get(ARGS, 1, "job_params.txt")
open(outpath, "w") do f
    for (N, t, U, Vpp, Vpm, op) in grid()
        println(f, "$N $t $U $Vpp $Vpm $op")
    end
end
println("Wrote $(length(grid())) jobs to $outpath")
