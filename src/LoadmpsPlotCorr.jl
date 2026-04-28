using Spin1LongRangeOrder
using ITensors
using Plots

function LoadmpsPlotCorr(filename::String; savefile::String="", titlestring::String="")
  # Load the MPS from the HDF5 file
  psi = load_simulation(filename)
  
  # Define the sites and the correlation function to plot
  N = length(psi)
  sites = siteinds(psi)
  
  # Compute the correlation function <S^z_i S^z_j> for a range of i and j
  zzcorr = correlation_matrix(psi, "Sz", "Sz")
  xxcorr = correlation_matrix(psi, "Sx", "Sx")
  yycorr = correlation_matrix(psi, "Sy", "Sy")

  xplotvals = range(start=1, length=N, step=1)
  line1 = abs.(zzcorr[Int64(N // 2), :])
  line2 = abs.(xxcorr[Int64(N // 2), :])
  line3 = abs.(yycorr[Int64(N // 2), :])

  # 2 panel figure with log-linear on top, log-log on bottom
  p1 = plot(xplotvals, line1, label="|ZZ|", xlabel="site", ylabel="Correlation", yscale=:log10, legend=:bottomleft,ms=5, lw=2)
  plot!(xplotvals, line2, label="|XX|", ms=5, lw=2)
  plot!(xplotvals, line3, label="|YY|", ms=5, lw=2)

  p2 = plot(xplotvals, line1, label="|ZZ|", xlabel="site", ylabel="Correlation", xscale=:log10, yscale=:log10, legend=:bottomleft,ms=5, lw=2)
  plot!(xplotvals, line2, label="|XX|", ms=5, lw=2)
  plot!(xplotvals, line3, label="|YY|", ms=5, lw=2)     

  plot!(p1, minorgrid=true)
  plot!(p2, minorgrid=true)
  p = plot(p1, p2, layout=(2,1), size=(800,600))

  title!(p, titlestring)

  if savefile != "" 
    savefig(p, savefile)
    println("Saved plot to ", savefile)
  else
    savefig(p, "temp_figure.png")
    println("No savefile provided, showing plot instead")
    gui(p)
  end
end

"""
    LoadmpsPlotCorr(filename::String, ::Val{:single}; savefile::String="", titlestring::String="")

Plot all correlations (Sz-Sz, Sx-Sx, Sy-Sy) with log-linear axes as a full figure.
Usage: LoadmpsPlotCorr("file.h5", Val(:single); savefile="output.png")
"""
function LoadmpsPlotCorr(filename::String, ::Val{:single}; savefile::String="", titlestring::String="")
  # Load the MPS from the HDF5 file
  psi = load_simulation(filename)
  
  # Define the sites and the correlation function to plot
  N = length(psi)
  sites = siteinds(psi)
  
  # Compute the correlation function
  zzcorr = correlation_matrix(psi, "Sz", "Sz")
  xxcorr = correlation_matrix(psi, "Sx", "Sx")
  yycorr = correlation_matrix(psi, "Sy", "Sy")

  xplotvals = range(start=1, length=N, step=1)
  line1 = abs.(zzcorr[Int64(N // 2), :])
  line2 = abs.(xxcorr[Int64(N // 2), :])
  line3 = abs.(yycorr[Int64(N // 2), :])

  # Single panel figure with log-linear axes
  p = plot(xplotvals, line1, label="|ZZ|", xlabel="site", ylabel="Correlation", 
           yscale=:log10, legend=:bottom, ms=5, lw=2, size=(800,500))
  plot!(xplotvals, line2, label="|XX|",  ms=5, lw=2)
  plot!(xplotvals, line3, label="|YY|", ms=5, lw=2)
  title!(p, titlestring)

  if savefile != "" 
    savefig(p, savefile)
    println("Saved plot to ", savefile)
  else
    savefig(p, "temp_figure.png")
    println("No savefile provided, showing plot instead")
    gui(p)
  end
end