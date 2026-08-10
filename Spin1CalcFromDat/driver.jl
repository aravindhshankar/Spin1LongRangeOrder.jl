include(joinpath(@__DIR__, "correlations.jl")) 

function all_datafiles()
    files = String[]
    for N in [16, 32, 64, 100, 128, 256, 512]
        for g1 in [-0.2]
            for g2 in collect(0.1:0.01:0.35)
                datafile = filename_builder(N, g1, g2)
                if isfile(datafile)
                    push!(files, datafile)
                end
            end
        end
    end
    return files
end

function main()
    files = all_datafiles()
    nfiles = length(files)

    task_id = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "0"))  # 0-indexed, 0..4
    ntasks  = parse(Int, get(ENV, "SLURM_ARRAY_TASK_COUNT", "5"))

    # Round-robin: this task handles files task_id, task_id+ntasks, task_id+2*ntasks, ...
    my_indices = (task_id + 1):ntasks:nfiles   # +1 because Julia is 1-indexed
    my_files = files[my_indices]

    println("Task $task_id / $ntasks handling $(length(my_files)) of $nfiles files")
    println("Threads.nthreads() = ", Threads.nthreads())

    for datafilename in my_files
        println("---- Task $task_id processing $datafilename ----")
        blas_threads_per_task = parse(Int, get(ENV, "MKL_NUM_THREADS", "1"))
        try
            compute_and_save_correlations(datafilename; blas_threads_per_task=blas_threads_per_task)
        catch e
            println("ERROR processing $datafilename: $e")
            # continue to next file rather than killing the whole array task
        end
    end
end

main()