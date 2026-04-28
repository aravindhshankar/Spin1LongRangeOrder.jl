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
    loadfile = "converged_g=$(g)N=100.h5"
    savefile = "g=$(g)N=100.png"
    LoadmpsPlotCorr(loadfile, Val(:single); savefile=savefile, titlestring="g=$g")
end
