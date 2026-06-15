using ITensors, ITensorMPS
using Printf
""" 
Compute variance of the Hamiltonian H in state psi: Var(H) = <H^2> - <H>^2. 
Check at the end of DMRG to ensure convergence to an eigenstate. 
"""
function variance_gs(H, psi)
  H2 = inner(H, psi, H, psi)   # Computes <H^2>
  E = inner(psi', H, psi)      # Computes <H>
  var = H2 - E^2               # Computes Variance
  return var
end

""" Compute the von Neumann entanglement entropy S = -Tr(ρ log ρ) across bond b. """
function entanglement_entropy(psi, b)
  psi = orthogonalize(psi, b)
  U,S,V = svd(psi[b], (linkinds(psi, b-1)..., siteinds(psi, b)...))
  SvN = 0.0
  for n=1:dim(S, 1)
    p = S[n,n]^2
    SvN -= p * log(p)
  end
return SvN
end

""" Show a command line bar chart of the entanglement entropy across all bonds. """
function entanglement_profile(psi)
    N = length(psi)
    println("\n  Entanglement entropy S(bond):")
    for b in 1:(N-1)
        s   = entanglement_entropy(psi, b)
        bar = repeat("█", round(Int, s * 10))
        @printf("  bond %2d–%2d:  S = %.4f  %s\n", b, b+1, s, bar)
    end
end

function ret_maxlinkdim(psi)
  return maximum([linkdim(psi, i) for i in 1:length(psi)-1])
end