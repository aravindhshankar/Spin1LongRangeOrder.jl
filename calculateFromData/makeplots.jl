using HDF5
using Plots
using Polynomials
using Printf
using LaTeXStrings
using FFTW
gr()

include(joinpath(@__DIR__, "correlations.jl"))  # for corr_output_filename, filename_builder, load_correlations

function plot_correlations(datafilename; site::Int, title_prefix="", nfit::Int=100)
    data = load_correlations(datafilename)
    @show keys(data)
    N = size(data.szmat, 1)
    xvals = 1:N

    # connected correlators built here, now that site is chosen
    szcorr_conn     = data.szmat[site, :]     .- data.sz_expect[site]   .* data.sz_expect
    chargecorr_conn = data.chargemat[site, :] .- data.ntot_expect[site] .* data.ntot_expect

    dist = xvals .- site
    keep = dist .> 0

    panels = [
        ("Sz-Sz (full)",          data.szmat[site, :]),
        ("Sz-Sz (connected)",     szcorr_conn),
        ("Ntot-Ntot (full)",      data.chargemat[site, :]),
        ("Ntot-Ntot (connected)", chargecorr_conn),
        ("S+ - S-",               data.spmmat[site, :]),
        ("Cdagup-Cup",            data.elecupmat[site, :]),
        ("Cdagdn-Cdn",            data.elecdnmat[site, :]),
    ]

    plts = Plots.Plot[]
    for (label, y) in panels
        xv = dist[keep]
        yv = abs.(y[keep])

        posmask = yv .> 0
        xv_pos, yv_pos = xv[posmask], yv[posmask]

        n_use = min(nfit, length(xv_pos))
        p = plot(xv, yv,
            xscale=:linear, yscale=:log10,
            xlabel="site - $site", ylabel="|correlation|",
            title=label, lw=2, ms=4, marker=:circle,
            label="data", legendfontsize=10)

        if n_use >= 2
            xfit = xv_pos[1:n_use]
            yfit = yv_pos[1:n_use]
            slope, intercept = loglog_fit(xfit, yfit)
            yline = 10 .^ (slope .* log10.(xfit) .+ intercept)
            fitlabel = @sprintf("slope=%.2f", slope)
            plot!(p, xfit, yline, ls=:dash, lw=2, color=:red, label=fitlabel)
        end

        push!(plts, p)
    end

    fullplot = plot(plts..., layout=(4, 2), size=(1100, 1500),
        plot_title="$title_prefix (ref site $site)")
    display(fullplot)
    return fullplot
end

##
let 
    N = 256
    U = 0.1
    Vpp = 0.8
    dV = 3.25 #choices 3.2, 3.5, 3.55, 3.7
    t = 1.0
    datafilename = filename_builder(N, t, U, Vpp, dV)

    plot_correlations(datafilename; site=Int(N//2), title_prefix="N=$N, U=$U, Vpp=$Vpp, dV=$dV", nfit=50)
end

##
let 
    #### PARA CORRELATION FUNCTIONS ############
    p = plot()
    colordict = Dict(64=>:blue, 128=>:green, 256=>:red)
    maxdist = range(1, stop=128, length=128)
    for N in (64, 128, 256)
        U = 0.1
        Vpp = 0.8
        dV = 3.8 #choices 3.2, 3.5, 3.55, 3.7
        t = 1.0
        datafilename = filename_builder(N, t, U, Vpp, dV)

        data = load_correlations(datafilename)
        site = Int(N//2)
        xvals = 1:N
        chargecorr_conn = data.chargemat[site, :] .- data.ntot_expect[site] .* data.ntot_expect
        @show length(chargecorr_conn)
        dist = xvals .- site
        keep = dist .> 0
        keep_neg = keep #.& (chargecorr_conn .< 0)
        @show length(chargecorr_conn[keep_neg])
        plot!(dist[keep_neg], abs.(-chargecorr_conn[keep_neg]), marker=:circle, 
        color=colordict[N], strokecolor=colordict[N], lw=2, ms=2, label="N=$N")
    end
    plot!(yscale=:log10, xscale=:log10)
    plot!(xlabel=raw"$x-\frac{N}{2}$", ylabel=raw"$| \delta\rho(x)\delta\rho(N/2)|$", title="Connected charge correlations", legendfontsize=12)
    # fitline = @. -0.21/ (pi * maxdist^2) + 0.01  * cos(0.0 * pi * maxdist) * maxdist^(-1 - 0.21*pi) / (log(maxdist))^1.5
    fitline = @. 0.07 * maxdist^(-2)
    plot!(maxdist, abs.(fitline), ls=:dash, lw=2, color=:black, label=raw"$\sim x^{-2}$")
    # plot!(maxdist, 0.0034 * maxdist.^(-1.21), ls=:dash, lw=2, color=:black, label=raw"$\sim x^{1.21}$")
end

##
let
    ######### PARA FOURIER ##############
    p = plot()
    colordict = Dict(64=>:blue, 128=>:green, 256=>:red)
    # omegaGlob = 2π .* range(0, 0.5, length=20)
    omegaGlob = 2π .* range(0, 0.05, length=20)
    for N in (64, 128, 256)
        U = 0.1
        Vpp = 0.8
        dV = 3.4 #choices 3.2, 3.5, 3.55, 3.7
        t = 1.0
        datafilename = filename_builder(N, t, U, Vpp, dV)

        data = load_correlations(datafilename)
        site = Int(N//2)
        xvals = 1:N
        chargecorr_conn = data.chargemat[site, :] .- data.ntot_expect[site] .* data.ntot_expect
        spinzcorr_conn = data.szmat[site, :] .- data.sz_expect[site] .* data.sz_expect
        electron_corr = 1 * (data.elecdnmat[site, :] ) 
        # spinpmcorr = data.spmmat[site, :]
        # chargecorr_conn = spinpmcorr
        # chargecorr_conn = spinzcorr_conn
        chargecorr_conn = electron_corr
        @show length(chargecorr_conn)
        dist = xvals .- site
        keep = dist .> 0
        keep_neg = keep #.& (chargecorr_conn .< 0)
        keep_neg_trunc = keep_neg #.& (dist .< Int(N//4))
        fft_chargecorr = fft(chargecorr_conn)
        omega = 2π * (0:N-1) / N
        # omega = 1:N-1
        @show length(chargecorr_conn[keep_neg_trunc])
        plot!(omega, abs.(fft_chargecorr), marker=:circle, 
        color=colordict[N], strokecolor=colordict[N], lw=2, ms=2, label="N=$N")
    end
    plot!(yscale=:linear, xscale=:linear)
    fitval = 0.21
    # fitval = 0.9/pi
    plot!(xlabel=raw"$k_n$", ylabel=raw"$ \mathcal{F}(\delta\rho(x)\delta\rho(N/2))$", title="Connected charge correlations", legendfontsize=12)
    plot!(omegaGlob, fitval .* omegaGlob, ls=:dash, lw=2, color=:black, label=label = L"$%$(fitval)\,\omega$")
    vline!([pi/2], ls=:dash, lw=2, color=:black, label=L"$\frac{\pi}{2}$")


end

##
let 
    ############# FREE FERMIONS FOURIER TRANSFORM for KC = 1 #############
    p = plot()
    colordict = Dict(64=>:blue, 128=>:green, 256=>:red)
    omegaGlob = 2π .* range(0, 0.5, length=20)
    N=32
    U = 0.0
    Vpp = 0.0
    dV = 0.0 #choices 3.2, 3.5, 3.55, 3.7
    t = 1.0
    site = Int(N//2)
    datafilename = "FF_N32_Npart16_t1.0.h5"
    psi = load_simulation(datafilename)
    chargecorr  = correlation_matrix(psi, "Ntot", "Ntot")
    chargecorr_conn = chargecorr[Int(N//2), :] .- expect(psi, "Ntot")[Int(N//2)] .* expect(psi, "Ntot")
    spinzcorr_conn = correlation_matrix(psi, "Sz", "Sz")[Int(N//2), :] .- expect(psi, "Sz")[Int(N//2)] .* expect(psi, "Sz")
    chargecorr_conn = spinzcorr_conn
    # electron_corrdn = correlation_matrix(psi, "Cdagdn", "Cdn")[Int(N//2), :]
    # electron_corrup = correlation_matrix(psi, "Cdagup", "Cup")[Int(N//2), :]
    # chargecorr_conn = (electron_corrdn .+ electron_corrup)
    xvals = 1:N

    @show length(chargecorr_conn)
    dist = xvals .- site
    keep = dist .> 0
    keep_neg = keep .& (chargecorr_conn .< 0)
    keep_neg_trunc = keep_neg .& (dist .< Int(N//4))
    fft_chargecorr = fft(chargecorr_conn)
    omega = 2π * (0:N-1) / N
    # omega = 1:N-1
    @show length(chargecorr_conn[keep_neg_trunc])
    plot!(omega, abs.(fft_chargecorr), marker=:circle, 
    color=:red, strokecolor=:red, lw=2, ms=2, label="N=$N")
    
    plot!(yscale=:linear, xscale=:linear)
    # fitval = 1.0/pi
    fitval = 0.25/pi # for spin correlations
    plot!(xlabel=raw"$k_n$", ylabel=raw"$ \mathcal{F}(\delta\rho(x)\delta\rho(N/2))$", title="Connected charge correlations in Free fermions", legendfontsize=12)
    plot!(omegaGlob, fitval .* omegaGlob, ls=:dash, lw=2, color=:black, label=label = L"$\frac{1}{\pi}\,k$")
end

##
let 
    ########## FREE FERMION CORRELATION FUNCTIONS ###################
    p = plot()
    colordict = Dict(64=>:blue, 128=>:green, 256=>:red)
    omegaGlob = 2π .* range(0, 0.5, length=20)
    maxdist = range(1, 32)
    N=32
    U = 0.0
    Vpp = 0.0
    dV = 0.0 #choices 3.2, 3.5, 3.55, 3.7
    t = 1.0
    site = Int(N//2)
    datafilename = "FF_N32_Npart16_t1.0.h5"
    psi = load_simulation(datafilename)
    # chargecorr  = correlation_matrix(psi, "Ntot", "Ntot")
    # chargecorr_conn = chargecorr[Int(N//2), :] .- expect(psi, "Ntot")[Int(N//2)] .* expect(psi, "Ntot")
    # spinzcorr_conn = correlation_matrix(psi, "Sz", "Sz")[Int(N//2), :] .- expect(psi, "Sz")[Int(N//2)] .* expect(psi, "Sz")
    # chargecorr_conn = spinzcorr_conn
    # electron_corrdn = correlation_matrix(psi, "Cdagdn", "Cdn")[Int(N//2), :]
    # electron_corrup = correlation_matrix(psi, "Cdagup", "Cup")[Int(N//2), :]
    chargecorr_conn = correlation_matrix(psi, "S+", "S-")[Int(N//2), :]
    xvals = 1:N

    @show length(chargecorr_conn)
    dist = xvals .- site
    keep = dist .> 0
    keep_neg = keep #.& (chargecorr_conn .< 0)
    keep_neg_trunc = keep_neg #.& (dist .< Int(N//4))

    @show length(chargecorr_conn[keep_neg_trunc])
    plot!(dist[keep_neg][1:1:end], abs.(chargecorr_conn[keep_neg][1:1:end]), marker=:circle, 
        color=:red, strokecolor=:red, lw=2, ms=2, label="N=$N")
    
    plot!(yscale=:log, xscale=:log)
    fitval = 1.0/pi
    fitval = 0.07
    fitval = 0.05
    plot!(xlabel=raw"$x-\frac{N}{2}$", ylabel=raw"$| \delta\rho(x)\delta\rho(N/2)|$", title=L"Connected charge correlations Free fermions ($2k_F = \frac{\pi}{2}$)", legendfontsize=12)
    plot!(maxdist, abs.(fitval * sin.(0.25 *pi .* maxdist)) .* maxdist.^(-2), ls=:dash, lw=2, color=:black, label=raw"$\sim x^{-2}$")
    plot!(maxdist, abs.(fitval * maxdist.^(-2)), ls=:dash, lw=2, color=:black, label=raw"$\sim \sin(2k_F x) / x$")
    plot!(ylims=[1e-7, 1])



end


let
    ######### PARA FOURIER +- corr ##############
    p = plot()
    colordict = Dict(64=>:blue, 128=>:green, 256=>:red)
    # omegaGlob = 2π .* range(0, 0.5, length=20)
    omegaGlob = 2π .* range(0, 0.05, length=20)
    for N in (64, 128, 256)
        U = 0.1
        Vpp = 0.8
        dV = 3.9 #choices 3.2, 3.5, 3.55, 3.7, 3.9
        t = 1.0
        datafilename = filename_builder(N, t, U, Vpp, dV)

        data = load_correlations(datafilename)
        site = Int(N//2)
        xvals = 1:N
        chargecorr_conn = data.chargemat[site, :] .- data.ntot_expect[site] .* data.ntot_expect
        spinzcorr_conn = data.szmat[site, :] .- data.sz_expect[site] .* data.sz_expect
        spinpmcorr = data.spmmat[site, :]
        chargecorr_conn = spinpmcorr
        @show length(chargecorr_conn)
        dist = xvals .- site
        keep = dist .> 0
        keep_neg = keep #.& (chargecorr_conn .< 0)
        keep_neg_trunc = keep_neg .& (dist .< Int(N//4))
        fft_chargecorr = fft(chargecorr_conn)
        omega = 2π * (0:N-1) / N
        # omega = 1:N-1
        @show length(chargecorr_conn[keep_neg_trunc])
        plot!(omega, abs.(fft_chargecorr), marker=:circle, 
        color=colordict[N], strokecolor=colordict[N], lw=2, ms=2, label="N=$N")
    end
    plot!(yscale=:linear, xscale=:linear)
    # fitval = 0.21
    fitval = 0.9/pi
    plot!(xlabel=raw"$k_n$", ylabel=raw"$ |\mathcal{F}(S^+(x)S^-(N/2))|$", title=L"Spin +- correlation ($2k_F = \pi/2$)", legendfontsize=12)
    # plot!(omegaGlob, fitval .* omegaGlob, ls=:dash, lw=2, color=:black, label=label = L"$%$(fitval)\,\omega$")
    vline!([pi/2, 3pi/2], ls=:dash, lw=2, color=:black, label=L"$\frac{\pi}{2}, \frac{3\pi}{2}$")

end


let 
    #### PARA SPM +- CORRELATION FUNCTIONS ############
    p = plot()
    colordict = Dict(64=>:blue, 128=>:green, 256=>:red)
    maxdist = range(1, stop=128, length=128)
    for N in (64, 128, 256)
        U = 0.1
        Vpp = 0.8
        dV = 3.65 #choices 3.2, 3.5, 3.55, 3.7
        t = 1.0
        datafilename = filename_builder(N, t, U, Vpp, dV)

        data = load_correlations(datafilename)
        site = Int(N//2)
        xvals = 1:N
        # chargecorr_conn = data.chargemat[site, :] .- data.ntot_expect[site] .* data.ntot_expect
        spinpmcorr = data.spmmat[site, :]
        chargecorr_conn = spinpmcorr
        @show length(chargecorr_conn)
        dist = xvals .- site
        keep = dist .> 0
        keep_neg = keep #.& (chargecorr_conn .< 0)
        @show length(chargecorr_conn[keep_neg])
        plot!(dist[keep_neg], abs.(-chargecorr_conn[keep_neg]), marker=:circle, 
        color=colordict[N], strokecolor=colordict[N], lw=2, ms=2, label="N=$N")
    end
    plot!(yscale=:log10, xscale=:log10)
    plot!(xlabel=raw"$x-\frac{N}{2}$", ylabel=raw"$| S^+(x)S^-(N/2)|$", title="Spin +- correlations", legendfontsize=12)
    # fitline = @. -0.21/ (pi * maxdist^2) + 0.01  * cos(0.0 * pi * maxdist) * maxdist^(-1 - 0.21*pi) / (log(maxdist))^1.5
    fitline = @. 0.07 * maxdist^(-0.947) #uses Kc = 0.67, Ks = 3.6
    plot!(maxdist, abs.(fitline), ls=:dash, lw=2, color=:black, label=raw"$\sim x^{-\,(K_c + \frac{1}{K_s})}$")
    plot!(legend =:bottomleft)
    # plot!(maxdist, 0.0034 * maxdist.^(-1.21), ls=:dash, lw=2, color=:black, label=raw"$\sim x^{1.21}$")
end