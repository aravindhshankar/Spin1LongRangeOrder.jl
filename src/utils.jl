using ITensors, ITensorMPS
function variance_gs(H, psi)
  H2 = inner(H, psi, H, psi)   # Computes <H^2>
  E = inner(psi', H, psi)      # Computes <H>
  var = H2 - E^2               # Computes Variance
  return var
end