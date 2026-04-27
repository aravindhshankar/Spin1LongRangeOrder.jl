using MKL
using ITensors, ITensorMPS
using HDF5
using Spin1LongRangeOrder

let
  N = 10
  sites = siteinds("S=1", N, conserve_qns=false)
  # psi = randomMPS(sites; linkdims=5)
  istate = ["Up" for i in 1:N]
  psi = randomMPS(sites, istate; linkdims=5)

  save_simulation("tempfile.h5", psi)
  psi_rec = load_simulation("tempfile.h5")
  
  println(typeof(psi_rec))
  @show mps_equal(psi, psi_rec)
  @show mps_tensor_equal(psi, psi_rec)






end #let
