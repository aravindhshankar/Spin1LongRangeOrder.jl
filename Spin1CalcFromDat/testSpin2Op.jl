using MKL
using Spin1LongRangeOrder
using ITensors, ITensorMPS
using Spin1LongRangeOrder.Hamiltonians
using HDF5
using LinearAlgebra
using Printf

##
let
    dataroot = "data/"
    sampledatafile = joinpath(dataroot, "N64/g1_-0.200_g2_0.190.h5")
    psi, params = load_simulation(sampledatafile, Val(:params))
    @show params
end

