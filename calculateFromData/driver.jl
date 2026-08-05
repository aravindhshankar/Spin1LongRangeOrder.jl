include(joinpath(@__DIR__, "correlations.jl")) 

function all_datafiles()
    files = String[]
    for N in [16,32,64,128,256]
        for U in [0.1]
            for Vpp in [0.8]
                for dV in 3.2:0.05:4.0  
                    for prefix in ["data/Hubbard/", "data/Hubbard/ImpPrec/"]
                        filename = filename_builder(N, 1.0, U, Vpp, dV; prefix=prefix, makepath=false)       # extend as needed, e.g. dV in 0.5:0.5:5.0
                        if isfile(filename)
                            push!(files, filename)
                        end
                    end
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
            compute_and_save_correlations(datafilename; blas_threads_per_task=blas_threads_per_task, skipout=false)
        catch e
            println("ERROR processing $datafilename: $e")
            # continue to next file rather than killing the whole array task
        end
    end
end

main()