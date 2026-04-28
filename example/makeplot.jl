using MKL
using Spin1LongRangeOrder
using ITensors, ITensorMPS
using Spin1LongRangeOrder.Hamiltonians
using HDF5
using Plots
gr()
## 
let 
    g=0.5
    N=100
    loadfile = "converged_g=$(g)N=$(N).h5"
    savefile = "g=$(g)N=$(N).png"
    LoadmpsPlotCorr(loadfile, Val(:single); savefile=savefile, titlestring="g=$g")
end
