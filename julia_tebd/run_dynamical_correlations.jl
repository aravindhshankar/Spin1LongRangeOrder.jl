using ITensors, ITensorMPS, HDF5
using Spin1LongRangeOrder # load_simulation, ret_maxlinkdim
include(joinpath(@__DIR__, "..", "calculateFromData", "correlations.jl"))  # filename_builder
include(joinpath(@__DIR__, "dynamical_correlation_tebd.jl"))

const DATAROOT = "data/Hubbard/tebd/"   

const DT                = 0.02
const CUTOFF             = 1e-10
const MAXDIM              = 700
const CHECKPOINT_EVERY   = 10   # save phi every 10 steps (t-spacing 0.2) so a crash loses little
const SNAPSHOT_EVERY     = 5   # save C(x,t) at the same cadence; cheap relative to the TEBD step itself

function main()
    N        = parse(Int, ARGS[1])
    t        = parse(Float64, ARGS[2])
    U        = parse(Float64, ARGS[3])
    Vpp      = parse(Float64, ARGS[4])
    Vpm      = parse(Float64, ARGS[5])
    operator = Symbol(ARGS[6])
    tf       = parse(Float64, ARGS[7])

    filename = filename_builder(N, t, U, Vpp, Vpm)
    tag = basename(filename)

    tempdir   = joinpath(DATAROOT, "temp_dyn")
    resultdir = joinpath(DATAROOT, "dyn_corr")
    mkpath(tempdir)
    mkpath(resultdir)

    checkpointpath = joinpath(tempdir, "$(tag)__op=$(operator).h5")
    snapshotpath   = joinpath(resultdir, "$(tag)__op=$(operator).h5")

    nsteps_target = round(Int, tf / DT)
    if isfile(checkpointpath)
        step0, t0 = peek_checkpoint(checkpointpath)
        if step0 >= nsteps_target
            println("Already evolved to t=$t0 >= requested tf=$tf, nothing to do.")
            return
        end
    end

    psi0, params = load_simulation(filename, Val(:all))
    println("Loaded $filename, bond dim = ", ret_maxlinkdim(psi0))
    c = div(N, 2)

    dynamical_correlation_tebd(
        psi0, tf;
        t=t, U=U, Vpp=Vpp, Vpm=Vpm,
        operator=operator, x0=c,
        dt=DT, cutoff=CUTOFF, maxdim=MAXDIM,
        checkpoint_path=checkpointpath, checkpoint_every=CHECKPOINT_EVERY,
        snapshot_path=snapshotpath, snapshot_every=SNAPSHOT_EVERY,
    )

    println("Snapshots up to t=$tf saved in $snapshotpath")
    # checkpoint is kept (not deleted) so this same job can be extended to a larger tf later
end

main()
