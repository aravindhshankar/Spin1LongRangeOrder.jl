using MKL
using Spin1LongRangeOrder
using ITensors, ITensorMPS
using Spin1LongRangeOrder.Hamiltonians
using HDF5
using LinearAlgebra
using Printf
using Plots
using Base.Threads
using Profile
using FFTW
using LaTeXStrings

gr()
fmt(x) = replace(@sprintf(" % .3f", x), " " => "")

## 
N = 64
dataroot = "data/N$(N)/"
g2val = 0.35
datafile = joinpath(dataroot, "g1_$(fmt(-0.200))_g2_$(fmt(g2val)).h5")
psi, params = load_simulation(datafile, Val(:params))
sz = expect(psi, "Sz")
spsm = correlation_matrix(psi, "S+", "S-")
szcorr = correlation_matrix(psi, "Sz", "Sz")
szconn = szcorr .- sz * sz'

##
let 
    mid = N ÷ 2
    x = 1:(N - mid)
    sp_corr = abs.(spsm[mid, mid+1:end])
    sz_corr = abs.(szconn[mid, mid+1:end])

    guide = 0.5 .* x .^ (-1.7)
    p = plot(
        x, sp_corr;
        xscale = :log10,
        yscale = :log10,
        lw = 2,
        marker = :square,
        label = "|⟨S⁺S⁻⟩|",
        xlabel = "x",
    )
    plot!( p, x, sz_corr; lw = 2, marker = :circle, label = "|connected ⟨SᶻSᶻ⟩|",)
    plot!(p, x, guide; lw = 2, ls = :dash, color = :black)
    title!(p, "N = $N, g₂ = $g2val")
    display(p)

end
##  
let
    mid = N ÷ 2
    omegaGlob = 2π .* range(0, 0.05, length=20)
    fft_szconn = fft(szconn[mid,:])
    omega = 2π * (0:N-1) / N
    p = plot(omega, abs.(fft_szconn), marker=:circle, 
    color=:red, strokecolor=:red, lw=2, ms=2, label="N=$N")
    
    plot!(yscale=:linear, xscale=:linear)
    # fitval = 1.0/pi
    fitval = 4.0/pi # for spin correlations
    plot!(xlabel=raw"$k_n$", ylabel=raw"$ \mathcal{F}(S^z(x)S^z(N/2))$")
    plot!(omegaGlob, fitval .* omegaGlob, ls=:dash, lw=2, color=:black, label=label = L"$\frac{1}{\pi}\,k$")
end
##
let 
    nthreads(:interactive)
    using LinearAlgebra
    BLAS.get_num_threads()
end

##
let
    psi, params = load_simulation("converged_g=0.5N=100.h5", Val(:params))
    @show maximum(linkdims(psi))
    @show params
    sz = expect(psi, "Sz")
    spsm = correlation_matrix(psi, "S+", "S-")
    szcorr = correlation_matrix(psi, "Sz", "Sz")
    szconn = szcorr .- sz * sz'
    N = 100
    mid = N ÷ 2
    x = 1:(N - mid)
    sp_corr = abs.(spsm[mid, mid+1:end])
    sz_corr = abs.(szconn[mid, mid+1:end])

    guide = 0.3 .* x .^ (-2)
    p = plot(
        x, sp_corr;
        xscale = :log10,
        yscale = :linear,
        lw = 2,
        marker = :square,
        label = "|⟨S⁺S⁻⟩|",
        xlabel = "x",
    )
    plot!( p, x, sz_corr; lw = 2, marker = :circle, label = "|connected ⟨SᶻSᶻ⟩|",)
    plot!(p, x, guide; lw = 2, ls = :dash, color = :black)
    title!(p, "N = $N, g₂ = $g2val")
    display(p)
end
