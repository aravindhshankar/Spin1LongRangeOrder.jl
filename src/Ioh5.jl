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

"""
    replace_siteinds(psi::MPS, new_sites::Vector)

Replace the site indices of a loaded MPS with new site indices.
This is necessary when resuming DMRG from a saved checkpoint to ensure
the loaded MPS uses the same site index objects as the new Hamiltonian.

Usage:
    sites = siteinds("S=1", N; conserve_sz=false)
    H = build_hamiltonian(sites, ...)
    psi_load = load_simulation("checkpoint.h5")
    psi_load = replace_siteinds(psi_load, sites)
    energy, psi = dmrg(H, psi_load; ...)
"""
function replace_siteinds(psi::MPS, new_sites::Vector)
    psi_new = copy(psi)
    
    for j in 1:length(psi_new)
        old_inds = inds(psi_new[j])
        
        # Find and replace the site index
        new_inds = map(old_inds) do ind
            if hasplev(ind) && plev(ind) == 0  # site index (no prime level)
                return new_sites[j]
            else
                return ind
            end
        end
        
        psi_new[j] = ITensor(psi_new[j], new_inds)
    end
    
    return psi_new
end


