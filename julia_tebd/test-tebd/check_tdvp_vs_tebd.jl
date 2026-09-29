# Small-N self-check: TDVP vs TEBD, C(x,t) vs a direct overlap, checkpoint resume.
# Run from the repo root: julia --project=. julia_tebd/test-tebd/check_tdvp_vs_tebd.jl
include(joinpath(@__DIR__, "..", "dynamical_correlation_tdvp.jl"))
using Random; Random.seed!(1)

N, t, U, Vpp, Vpm = 12, 1.0, 0.1, 0.8, 4.0
s = siteinds("Electron", N; conserve_nf=true)
H = build_hamiltonian(s, t, U, Vpp, Vpm)
E0, psi0 = dmrg(H, random_mps(s, [isodd(j) ? "Up" : "Dn" for j in 1:N]; linkdims=10);
                nsweeps=10, maxdim=[20, 50, 100, 200], cutoff=1e-12, outputlevel=0)
kw = (t=t, U=U, Vpp=Vpp, Vpm=Vpm, cutoff=1e-12, maxdim=200, verbose=false, E0=E0)

for op in (:Sz, :cup)
    opname, opdag = _op_names(op)
    # t = 0: must equal the equal-time <O†(x) O(x0)> from ITensors' correlation_matrix
    phi0 = apply_local_operator(psi0, opname, N ÷ 2)   # same construction as the driver
    C0 = local_operator_correlation(psi0, phi0, opdag, s)
    e0 = maximum(abs, C0 .- correlation_matrix(psi0, opdag, opname)[:, N ÷ 2])

    tebd1 = dynamical_correlation_tebd(psi0, 1.0; operator=op, dt=0.02, kw...)
    tebd2 = dynamical_correlation_tebd(psi0, 1.0; operator=op, dt=0.01, kw...)
    tdvp  = dynamical_correlation_tdvp(psi0, 1.0; operator=op, dt=0.02, kw...)
    d1, d2 = maximum(abs, tdvp .- tebd1), maximum(abs, tdvp .- tebd2)

    ck = tempname() * ".h5"
    dynamical_correlation_tdvp(psi0, 0.5; operator=op, dt=0.02, checkpoint_path=ck, kw...)
    resumed = dynamical_correlation_tdvp(psi0, 1.0; operator=op, dt=0.02, checkpoint_path=ck, kw...)
    dr = maximum(abs, resumed .- tdvp)

    println("$op: |C(t=0) - direct| = $e0  |tdvp - tebd(dt=.02)| = $d1  |tdvp - tebd(dt=.01)| = $d2  ",
            "|resumed - straight| = $dr  max|C| = $(maximum(abs, tdvp))")
    @assert e0 < 1e-10
    @assert d2 < d1 < 1e-3      # TEBD converges towards TDVP as its Trotter error shrinks
    @assert dr < 1e-8
end
@assert abs(E0 - real(inner(psi0, H, psi0))) < 1e-8
println("all checks passed")
