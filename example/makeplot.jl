using MKL
using Spin1LongRangeOrder
using ITensors, ITensorMPS
using Spin1LongRangeOrder.Hamiltonians
using HDF5
using Plots
using Printf
gr()
## 
# let 
#     g=0.5
#     N=100
#     loadfile = "converged_g=$(g)N=$(N).h5"
#     savefile = "g=$(g)N=$(N).png"
#     LoadmpsPlotCorr(loadfile, Val(:single); savefile=savefile, titlestring="g=$g")
# end

## 
function filename_builder(N, t, U, Vpp, dV)
    Vpm = Vpp + dV
    Npart = Int(N // 2)
    _ = t # not used, so we discard, but leave the API as is for the future
    datasavedir = "data/Hubbard/N$N" * "consNf/"
    datafilename = datasavedir * "N$N" * "_U" * @sprintf("%.3f", U) * "_Vpp" * @sprintf("%.3f", Vpp) * "_Vpm" * @sprintf("%.3f", Vpm) * "_Np$Npart" * raw".h5"
    return datafilename
end

N = 256
Nf = 128
Vpp = 0.8
U = 0.1 
dV = 3.2
Vpm = Vpp + dV
t = 1.0
datafilename = filename_builder(N, t, U, Vpp, dV)
psi, params = load_simulation(datafilename, Val(:all))
println("Loaded simulation from $datafilename")
println("params = ", params)
println("params keys = ", collect(keys(params)))
szcorr = correlation_matrix(psi, "Sz", "Sz")[20, :]
spmcorr = correlation_matrix(psi, "S+", "S-")[20, :]
chargecorr = correlation_matrix(psi, "Ntot", "Ntot")[20, :]
electroncorr = correlation_matrix(psi, "Cdagup", "Cup")[20, :]


##
let
yvals = abs.(szcorr)
xvals = 1:length(yvals)
p = plot(xvals, yvals, title="N=$N, Nf=$Nf, Vpm=$Vpm",
     xlabel="site", ylabel="SzSz", legend=false, 
     yscale=:log10, xscale=:log10, lw=2, ms=5)

display(p)
end
##