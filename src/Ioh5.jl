using ITensors, ITensorMPS
using HDF5
using Dates

# export save_simulation, load_simulation, mps_equal, mps_tensor_equal

"""
Usage : 
save_simulation("file.h5", psi)                          # just psi
save_simulation("file.h5", psi; params=Dict(...))        # only params
save_simulation("file.h5", psi; metadata=Dict(...))      # only metadata
"""
function save_simulation(filename::String, psi::MPS)
    h5open(filename, "w") do f
        write(f, "psi", psi)
    end
    return nothing
end

function save_simulation(filename::String, psi::MPS, params::Dict)
    h5open(filename, "w") do f
        write(f, "psi", psi)

        g = create_group(f, "params")
        for (k, v) in params
            write(g, string(k), v)
        end
    end
    return nothing
end

function save_simulation(
    filename::String,
    psi::MPS,
    params::Dict,
    metadata::Dict
)
    h5open(filename, "w") do f
        write(f, "psi", psi)

        g_params = create_group(f, "params")
        for (k, v) in params
            write(g_params, string(k), v)
        end

        g_meta = create_group(f, "metadata")
        for (k, v) in metadata
            write(g_meta, string(k), v)
        end
    end
    return nothing
end


# -------------------
# LOAD
# -------------------

# --- load only psi ---
function load_simulation(filename::String)
    return h5open(filename, "r") do f
        read(f, "psi", MPS)
    end
end


# --- load psi + params ---
function load_simulation(filename::String, ::Val{:params})
    return h5open(filename, "r") do f
        psi = read(f, "psi", MPS)

        params = Dict{String,Any}()
        if haskey(f, "params")
            for k in keys(f["params"])
                params[k] = read(f["params"], k)
            end
        end

        return psi, params
    end
end


# --- load everything ---
function load_simulation(filename::String, ::Val{:all})
    return h5open(filename, "r") do f
        psi = read(f, "psi", MPS)
        params = Dict{String,Any}()
        metadata = Dict{String,Any}()

        if haskey(f, "params")
            for k in keys(f["params"])
                params[k] = read(f["params"], k)
            end
        end

        if haskey(f, "metadata")
            for k in keys(f["metadata"])
                metadata[k] = read(f["metadata"], k)
            end
        end

        return psi, params, metadata
    end
end

"""
checks only innter product, could be in any gauge 
"""
function mps_equal(psi1::MPS, psi2::MPS; atol=1e-12)
    # align orthogonality center / gauge (optional but helpful)
    psi1n = normalize(psi1)
    psi2n = normalize(psi2)

    ov = inner(psi1n, psi2n)

    return abs(abs(ov) - 1) < atol
end

"""
Strict tensor by tensor checking 
"""
function mps_tensor_equal(psi1::MPS, psi2::MPS; atol=1e-12)
    length(psi1) == length(psi2) || return false

    for j in 1:length(psi1)
        if norm(psi1[j] - psi2[j]) > atol
            return false
        end
    end

    return true
end
