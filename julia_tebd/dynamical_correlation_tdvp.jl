using TenNetLib
include(joinpath(@__DIR__, "dynamical_correlation_tebd.jl"))

# -----------------------------------------------------------------------
# TDVP (TenNetLib) replacement for the TEBD stepper. nsite="dynamic" does
# 2-site updates on bonds still below maxdim and 1-site updates on the rest,
# so the bond dimension can grow without a separate subspace expansion.
# normalize must stay false by default: TenNetLib normalizes otherwise,
# which would drop ||O(x0) psi0|| from C(x,t).
# -----------------------------------------------------------------------
function tdvp_stepper(phi, H; sites, t, U, Vpp, Vpm, dt, cutoff, maxdim, normalize)
    H === nothing && (H = build_hamiltonian(sites, t, U, Vpp, Vpm))   # not built on checkpoint resume
    engine = TDVPEngine(phi, H)
    return function (_)
        # TenNetLib applies exp(time_step * H), so real time needs -im*dt
        tdvpsweep!(engine, -im * dt; nsite="dynamic", maxdim=maxdim, cutoff=Float64(cutoff),
                   normalize=normalize, outputlevel=0)
        return getpsi(engine)
    end
end

# Same arguments and return value as dynamical_correlation_tebd.
dynamical_correlation_tdvp(psi0::MPS, tf::Real; kwargs...) =
    dynamical_correlation_tebd(psi0, tf; stepper=tdvp_stepper, kwargs...)
