# Needs to be run from project root.
using Spin1LongRangeOrder
include(joinpath(@__DIR__, "..", "calculateFromData", "correlations.jl"))  # filename_builder
include(joinpath(@__DIR__, "dynamical_correlation_tebd.jl"))                       # peek_checkpoint
include(joinpath(@__DIR__, "params_grid.jl"))

const DATAROOT = "data/Hubbard/tebd/"   # <-- keep in sync with run_dynamical_correlations.jl
const DT = 0.02                        # <-- keep in sync with run_dynamical_correlations.jl

tf = parse(Float64, get(ARGS, 1, "1.0"))
outpath = get(ARGS, 2, "job_params_resume.txt")
nsteps_target = round(Int, tf / DT)

tempdir = joinpath(DATAROOT, "temp_dyn")

open(outpath, "w") do f
    n = 0
    for (N, t, U, Vpp, Vpm, op) in grid()
        tag = basename(filename_builder(N, t, U, Vpp, Vpm))
        checkpointpath = joinpath(tempdir, "$(tag)__op=$(op).h5")
        done = isfile(checkpointpath) && peek_checkpoint(checkpointpath)[1] >= nsteps_target
        if !done
            println(f, "$N $t $U $Vpp $Vpm $op")
            n += 1
        end
    end
    println("$n job(s) not yet at tf=$tf written to $outpath")
end
