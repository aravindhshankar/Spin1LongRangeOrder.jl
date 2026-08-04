using MKL
using Spin1LongRangeOrder
using ITensors, ITensorMPS
using Spin1LongRangeOrder.Hamiltonians
using HDF5
using LinearAlgebra
using Printf
using Plots
gr()
fmt(x) = replace(@sprintf(" % .3f", x), " " => "")

## 
let 
    dataroot = "data/N64/"
    g2val = 0.25
    datafile = joinpath(dataroot, "g1_$(fmt(-0.200))_g2_$(fmt(g2val)).h5")
    psi, params = load_simulation(datafile, Val(:params))

    p = plot(1:length(psi), expect(psi, "Sz"), label="Sz", xlabel="x", ylabel="<sz>")
    display(p)
    @show params
end