using PyPlot

using LinearAlgebra
using LsqFit
using Measurements
using StatsBase


function plot_initial_conditions(sim::Array{Sol,1})

    u0 = join_trajectories(sim,1)

    N::Int = sum(try sol.p.N catch; sol.p[10] end for sol in sim) # for backwards compatibility
    nbins = trunc(Int, sqrt(N))

    rangex = range(0,length=trunc(Int,sqrt(N)),stop=2pi)
    pdfx0 = fit(Histogram,mod2pi.(u0[1:N]),rangex)
    pdfx0 = normalize(pdfx0)

    minp = minimum(u0[N+1:2N])
    maxp = maximum(u0[N+1:2N])
    rangep = range(minp,length=trunc(Int,sqrt(N)),stop=maxp)
    pdfp0 = fit(Histogram,u0[N+1:2N],rangep)
    pdfp0 = normalize(pdfp0)

    ranges = range(-1.1,length=trunc(Int,sqrt(N)),stop=1.1)
    pdfsx = fit(Histogram,u0[2N+1:3N],ranges)
    pdfsy = fit(Histogram,u0[3N+1:4N],ranges)
    pdfsz = fit(Histogram,u0[4N+1:5N],ranges)
    pdfsx = normalize(pdfsx)
    pdfsy = normalize(pdfsy)
    pdfsz = normalize(pdfsz)


    matplotlib[:rc]("axes", labelpad=2.)

    fig, ax = subplots(2,2,figsize=[6.2, 4.6])

    ax[1, 1][:set_ylabel](L"P(x_j)")
    # ax[1, 1][:set_xlabel](L"atom position $x$")
    ax[1, 1][:set_ylim]([0.,maximum(pdfx0.weights)*1.1])
    ax[1, 1][:set_xticks](pi*collect(0:2))
    ax[1, 1][:set_xticklabels]([L"0",L"\pi",L"2\pi"])
    ax[1, 1][:step](pdfx0.edges[1][1:end-1],pdfx0.weights,label="initial",where="post", color="skyblue", lw=1.5)

    ax[1, 2][:set_ylabel](L"P(p_j)")
    # ax[1, 2][:set_xlabel](L"$p_j$")
    ax[1, 2][:step](pdfp0.edges[1][1:end-1],pdfp0.weights,label="initial",where="post", color="lightslategrey", lw=1.5)

    ax[2, 1][:set_ylabel](L"P(\sigma)")
    ax[2, 1][:set_xlabel](L"$\sigma^x$ and $\sigma^y$")
    ax[2, 1][:set_xlabel](L"$\sigma^x$ and $\sigma^y$")
    ax[2, 1][:step](pdfsx.edges[1][1:end-1],pdfsx.weights,label=L"\sigma^x",where="post", color="indianred", lw=1.5)
    ax[2, 1][:step](pdfsy.edges[1][1:end-1],pdfsy.weights,label=L"\sigma^y",where="post", color="sandybrown", lw=1.5)
    ax[2, 1][:legend]()

    ax[2, 2][:set_ylabel](L"P(\sigma^z)")
    ax[2, 2][:set_xlabel](L"spins $\sigma^z$")
    ax[2, 2][:step](pdfsz.edges[1][1:end-1],pdfsz.weights,label=L"\sigma^z",where="post", color = "mediumpurple", lw=1.5)


    fig[:tight_layout](h_pad=0., w_pad=-0.)

    return fig, ax
end



function plot_position(sim::Array{Sol,1},idx::Int=size(sim[1].t)[1])

    u0 = join_trajectories(sim,1)
    u1 = join_trajectories(sim,idx)

    N::Int = sum(try sol.p.N catch; sol.p[10] end for sol in sim) # for backwards compatibility
    nbins = trunc(Int, sqrt(N))

    rangex = range(0,length=trunc(Int,sqrt(N)),stop=1.)
    pdfx0 = fit(Histogram,mod2pi.(u0[1:N])./(2pi),rangex)
    pdfx1 = fit(Histogram,mod2pi.(u1[1:N])./(2pi),rangex)

    pdfx0 = normalize(pdfx0)
    pdfx1 = normalize(pdfx1)

    minp = minimum(vcat(u0[N+1:2N],u1[N+1:2N]))
    maxp = maximum(vcat(u0[N+1:2N],u1[N+1:2N]))
    rangep = range(minp,length=trunc(Int,sqrt(N)),stop=maxp)
    pdfp0 = fit(Histogram,u0[N+1:2N],rangep)
    pdfp1 = fit(Histogram,u1[N+1:2N],rangep)


    @. model(x, p) = p[1]*cos(x*p[2])+p[3]
    xdata = collect(pdfx1.edges[1][1:end-1])
    ydata = pdfx1.weights
    p0 = [600., 1., 0.]
    fitted = curve_fit(model, xdata, ydata, p0)
    println("Fitted parameters", fitted.param)
    matplotlib[:rc]("axes", labelpad=2)

    fig, ax = subplots(1,1,figsize=[6.8, 5.2])

    ax[:set_ylabel]("distribution")
    ax[:set_xlabel](L"atom position ($1/\lambda_\mathrm{c}$)")
    # ax[:set_xticks](pi/2*collect(0:4))
    # ax[:set_xticklabels]([L"0",L"\pi/2",L"\pi",L"3\pi/2",L"2\pi"])
    ax[:step](pdfx0.edges[1][1:end-1],pdfx0.weights,label="initial",where="post")
    ax[:step](pdfx1.edges[1][1:end-1],pdfx1.weights,label="final",where="post")
    # ax[:plot](xdata,model(xdata,fitted.param))
    ax[:legend](loc=8,ncol=1)

    fig[:tight_layout]()

    return fig, ax
end

# function plot_spinspositionhisto(sim::Array{Sol,1})
#     u1 = join_trajectories(sim,size(sim[1].t)[1])

#     N::Int = sum(try sol.p.N catch; sol.p[10] end for sol in sim) # for backwards compatibility

#     nbins = trunc(Int, ^(N,1//3))
#     println("number of bins: "*string(nbins))

#     matplotlib[:rc]("axes", labelpad=1)
#     matplotlib[:rc]("image", cmap="inferno")

#     x = mod2pi.(u1[1:N])/(2pi)

#     sf = 1.5
#     fig, ax = subplots(1,2,figsize=[4.25*sf, 2.0*sf],sharex=true,sharey=true)

#     ax[1][:set_ylabel](L"\langle S_x\rangle",labelpad=-5.)
#     ax[1][:hist2d](x,u1[2N+1:3N],bins=nbins,density=true)   #,norm=matplotlib[:colors][:LogNorm]())

#     ax[2][:set_ylabel](L"\langle S_z\rangle",labelpad=4.)
#     cs = ax[2][:hist2d](x,u1[4N+1:5N], bins=nbins,density=true)

#     for i in ax
#         # i[:set_xlabel](L"atom position mod $2\pi$ (units of $\lambda_\mathrm{c}$)")
#         i[:set_xticks](0:0.5:1)
#         # i[:set_xticklabels]([L"0",L"\pi/2",L"\pi",L"3\pi/2",L"2\pi"])
#         i[:set_yticks](collect(-1:0.5:1))
#         # i[:set_yticklabels]([L"-1",L"-\frac{1}{2}",L"0",L"\frac{1}{2}",L"1"])
#     end

#     fig[:tight_layout](h_pad=0.0, w_pad=-0.3)
#     fig[:text](0.5, 0.02, L"atom position (1/$\lambda_\mathrm{c}$)", ha="center")

#     fig[:subplots_adjust](right=0.86)
#     cbar_ax = fig[:add_axes]([0.88, 0.2, 0.02, 0.7])
#     cbar = fig[:colorbar](cs[4],cax=cbar_ax)
#     # cbar[:set_label](L"\langle a^\dag a\rangle")


#     return fig, ax

# end


function plot_spinspositionhisto(sim::Array{Sol,1}; cmap="inferno")
    if isempty(sim)
        error("No trajectories found in `sim`")
    end

    sol0 = sim[1]
    N = sol0.p.N
    t = sol0.t
    Nt = length(t)

    nbins = trunc(Int, ^(N,1//2))

    edges = collect(range(0.0, stop=2π, length=nbins+1))
    centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])   


    Sx_mat = fill(NaN, nbins, Nt)
    Sz_mat = fill(NaN, nbins, Nt)
    counts_mat = zeros(Int, nbins, Nt)

    
    for jj in 1:Nt
        pos_pool = Float64[]
        sx_pool  = Float64[]
        sz_pool  = Float64[]
        for sol in sim

            x = mod.(sol.u[1:N, jj], 2π)           # positions in [0,2π)
            sx = sol.u[(2N+1):(3N), jj]            # Sx per site
            sz = sol.u[(4N+1):(5N), jj]            # Sz per site

            append!(pos_pool, x)
            append!(sx_pool, sx)
            append!(sz_pool, sz)
        end

        
        if !isempty(pos_pool)

            h_counts = fit(Histogram, pos_pool, edges)
            counts = copy(h_counts.weights) .|> Int


            h_sx = fit(Histogram, pos_pool, Weights(sx_pool), edges)
            h_sz = fit(Histogram, pos_pool, Weights(sz_pool), edges)

            # MEANS
            for b in 1:nbins
                counts_mat[b, jj] = counts[b]
                if counts[b] > 0
                    Sx_mat[b, jj] = h_sx.weights[b] / counts[b]
                    Sz_mat[b, jj] = h_sz.weights[b] / counts[b]
                else
                    Sx_mat[b, jj] = NaN
                    Sz_mat[b, jj] = NaN
                end
            end
        end
    end


    sf = 1.5
    fig, ax = subplots(1,2, figsize=[4.25*sf, 2.0*sf], sharex=true, sharey=true)

    # extent: (x0, x1, y0, y1)
    extent = (t[1], t[end], 0.0, 2π)

    im1 = ax[1][:imshow](Sx_mat, origin="lower", aspect="auto", extent=extent, interpolation="nearest", cmap="RdBu_r",vmin=-0.5, vmax=0.5)
    ax[1][:set_title](L"\langle S_x\rangle")
    ax[1][:set_xlabel](L"t", labelpad=-7.5)
    ax[1][:set_ylabel](L"x/\lambda_c")

    im2 = ax[2][:imshow](Sz_mat, origin="lower", aspect="auto", extent=extent, interpolation="nearest", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax[2][:set_title](L"\langle S_z\rangle")
    ax[2][:set_xlabel](L"t", labelpad=-7.5)


    yticks_positions = [0.0, π, 2π]
    ax[1][:set_yticks](yticks_positions)
    ax[1][:set_yticklabels]([L"0",  L"\pi",  L"2\pi"])

    fig[:tight_layout](h_pad=0.0, w_pad=0.1)
    fig[:subplots_adjust](right=0.88)
    cbar_ax1 = fig[:add_axes]([0.90, 0.35, 0.02, 0.35])
    cbar1 = fig[:colorbar](im1, cax=cbar_ax1)

    ax[1].tick_params(direction="in", which="both")
    ax[2].tick_params(direction="in", which="both")

    cbar_ax1.tick_params(direction="in", which="both")

    display(fig)
    return fig, (Sx_mat, Sz_mat, counts_mat, t, centers)
end

function plot_contact_spatial_correlation(sim; bins=50, figsize=(8, 4), savefile=nothing)
    sols = extract_solution(sim)
    Ntrajs = length(sols)
    

    contact_distances = Float64[]
    contact_midpoints = Float64[]
    
    for m in 1:Ntrajs
        sol = sols[m]
        N = sol.p.N
        dressed = dressed_idx
        Nr = length(dressed)
        if Nr <= 1
            continue
        end
        rth = sol.p.rthresh
        
        for j in 1:length(sol.t)
            pos = sol.u[1:N, j]
            for a in 1:(Nr-1)
                xa = pos[dressed[a]]
                for b in (a+1):Nr
                    xb = pos[dressed[b]]
                    dx = abs(xa - xb)
                    dx = mod(dx, 2π)
                    if dx > π
                        dx = 2π - dx
                    end
                    if dx <= rth
                        push!(contact_distances, dx)
                        mid_pos = mod((xa + xb)/2, 2π)
                        push!(contact_midpoints, mid_pos)
                    end
                end
            end
        end
    end

    fig, (ax1, ax2) = subplots(1, 2, figsize=figsize)
    
    if !isempty(contact_distances)
        # Plot 1: Distribution of contact distances
        ax1.hist(contact_distances, bins=min(bins, 50), density=true, alpha=0.7, color="C0")
        ax1.axvline(rthresh, color="grey", linestyle="--", label=L"R_0")
        ax1.set_xlabel("Contact distance")
        ax1.set_ylabel(L"P(x)")
        ax1.legend()
        
        # Plot 2: Distribution of contact midpoints
        ax2.hist(contact_midpoints, bins=bins, density=true, alpha=0.7, color="C1")
        ax2.set_xlabel(L"$(x_i+x_j)/2$")
        ax2.set_ylabel(L"P(x)")
        ax2.set_xlim(0, 2π)
        ax2.set_xticks([0,π, 2π],[L"0", L"\lambda_c/2", L"\lambda_c"])
    else
        @warn "No contacts found to plot"
    end
    
    tight_layout()

    if savefile !== nothing
        savefig(savefile)
    end
    return fig
end


function plot_contact_count(sim; figsize=(10,4), show_individual=true, savefile=nothing)
    sols = extract_solution(sim)
    Ntrajs = length(sols)
    if Ntrajs == 0
        error("No trajectories found in `sim`.")
    end


    t_ref = sols[1].t
    Nt = length(t_ref)
    for s in sols
        if length(s.t) != Nt || any(s.t .!= t_ref)
            error("All trajectories must share the same time vector `t`.")
        end
    end

    # compute contact-count time-series for each trajectory 
    counts = zeros(Float64, Ntrajs, Nt)
    for m in 1:Ntrajs
        sol = sols[m]

        N = sol.p.N
        dressed = dressed_idx
        Nr = length(dressed)
        if Nr <= 1
            # leave zeros
            continue
        end
        rth = sol.p.rthresh
        @inbounds for j in 1:Nt
            pos = sol.u[1:N, j]
            cnt = 0
            for a in 1:(Nr-1)
                xa = pos[dressed[a]]
                for b in (a+1):Nr
                    xb = pos[dressed[b]]
                    dx = abs(xa-xb)
                    dx = mod(dx, 2π)
                    if dx > π
                        dx = 2π - dx
                    end
                    if dx <= rth
                        cnt += 1
                    end
                end
            end
            counts[m, j] = float(cnt)
        end
    end

    meancounts = mean(counts, dims=1)[:]   

    fig, ax = subplots(figsize=figsize)
    if show_individual

        grey_levels = range(0.3, 0.9, length=Ntrajs) 
        for m in 1:Ntrajs
            grey_val = grey_levels[m]
            plot(t_ref, counts[m, :], alpha=0.7, linewidth=0.8, color=(grey_val, grey_val, grey_val))
        end
    end
    plot(t_ref, meancounts, linewidth=2.2, color="C1", label="mean")
    xlabel("t")
    ylabel("contact count")
    # yticks(range(0,N,N+1))
    # legend(loc="best")
    # grid(true)
    tight_layout()

    if savefile !== nothing
        savefig(savefile)
    end
    return fig
end


###########################################

        #Wigner function specific#

###########################################



function get_cavity_samples(sim; time_idx::Union{Int,Nothing}=nothing)
    sols = sols_from_sim(sim)
    if isempty(sols)
        error("No trajectories found in sim")
    end

    Nt = length(sols[1].t)
    t_idx = time_idx === nothing ? Nt : time_idx
    Ntrajs = length(sols)
    Nsys = sols[1].p.N
    samples = zeros(Float64, Ntrajs, 2)
    for (i, sol) in enumerate(sols)
        # Sanity checks (fail early if inconsistent)
        if length(sol.t) < t_idx
            error("Trajectory $(i) has fewer time points than requested index $t_idx")
        end
        if sol.p.N != Nsys
            error("All trajectories must have the same sol.p.N")
        end
        # cavity quadratures stored in slots 5N+1 : 5N+2
        samples[i, 1] = sol.u[5*Nsys + 1, t_idx]   # a_x
        samples[i, 2] = sol.u[5*Nsys + 2, t_idx]   # a_p
    end
    return samples
end

# 2D Gaussian KDE (simple, robust). 
# - samples: Nx2 matrix
# - nbins: grid resolution
# - hscale: multiply Scott's-rule bandwidth by this factor (use <1 to sharpen, >1 to smooth)
function kde2d(samples::AbstractMatrix{<:Real}; nbins::Int=201, hscale::Float64=1.0)
    N, d = size(samples)
    @assert d == 2 "kde2d expects Nx2 sample array"

    # If too few samples, fall back to Gaussian-fit
    if N < 5
        μ = vec(mean(samples, dims=1))
        Σ = cov(samples; corrected=false) + 1e-12I(2)
        xs = range(μ[1] - 3*sqrt(Σ[1,1]), μ[1] + 3*sqrt(Σ[1,1]), length=nbins)
        ys = range(μ[2] - 3*sqrt(Σ[2,2]), μ[2] + 3*sqrt(Σ[2,2]), length=nbins)
        dx = step(xs); dy = step(ys)
        W = zeros(Float64, nbins, nbins)
        invΣ = inv(Σ)
        normconst = 1.0 / (2π * sqrt(det(Σ)))
        for ix in 1:nbins, iy in 1:nbins
            v = [xs[ix]; ys[iy]] .- μ
            W[iy, ix] = normconst * exp(-0.5 * (v' * invΣ * v))
        end
        W ./= (sum(W) * dx * dy)
        return xs, ys, W
    end

    # Scott's rule for d=2: h = sigma * N^(-1/(d+4))
    sig = vec(std(samples, dims=1))
    # ensure nonzero sigma
    sig[sig .== 0.0] .= 1e-8
    scotts = N ^ (-1.0 / (d + 4))
    hx = sig[1] * scotts * hscale
    hp = sig[2] * scotts * hscale


    xmin = minimum(samples[:, 1]) - 3hx
    xmax = maximum(samples[:, 1]) + 3hx
    ymin = minimum(samples[:, 2]) - 3hp
    ymax = maximum(samples[:, 2]) + 3hp

    xs = collect(range(xmin, stop=xmax, length=nbins))
    ys = collect(range(ymin, stop=ymax, length=nbins))
    dx = xs[2] - xs[1]
    dy = ys[2] - ys[1]


    norm = 1.0 / (2π * hx * hp * N)
    hx2 = hx^2
    hp2 = hp^2


    W = zeros(Float64, nbins, nbins)  # (iy, ix) -> (y, x)
    for ix in 1:nbins
        dxs = xs[ix] .- samples[:, 1]          
        dxs2 = (dxs.^2) ./ hx2
        for iy in 1:nbins
            dys = ys[iy] .- samples[:, 2]
            exponents = @. exp(-0.5 * (dxs2 + (dys^2)/hp2))
            W[iy, ix] = norm * sum(exponents)
        end
    end

    # Normalize 
    W ./= (sum(W) * dx * dy)

    return xs, ys, W
end

# Top-level: compute and plot cavity Wigner at a chosen time index (default: final time)
function plot_cavity_wigner(sim; time_idx::Union{Int,Nothing}=nothing, nbins::Int=201, hscale::Float64=1.0, cmap="RdBu_r", show_contours::Bool=true)
    samples = get_cavity_samples(sim; time_idx=time_idx)
    xs, ys, W = kde2d(samples; nbins=nbins, hscale=hscale)

    fig, ax = subplots(figsize=(6,5))
    extent = (first(xs), last(xs), first(ys), last(ys))
    im = ax[:imshow](W, origin="lower", extent=extent, aspect="auto", interpolation="bilinear",cmap="inferno")#,vmin=-maximum(abs.(W)), vmax=maximum(abs.(W)),cmap="pink")
    ax[:set_xlabel](L"a_x"); ax[:set_ylabel](L"a_p")
    cbar = fig[:colorbar](im, ax=ax)
    cbar[:set_label](L"W(a_x,a_p)")

    if show_contours
        levels = range(minimum(W), stop=maximum(W), length=7)[2:end-1]
        ax[:contour](xs, ys, W, levels=collect(levels), linewidths=1.0)
    end


    ax_x = fig[:add_axes]([0.2, 0.85, 0.50, 0.08]) 
    ax_x[:plot](xs, sum(W, dims=1)[:] .* (ys[2]-ys[1]))  # marginal along x
    ax_x[:set_xticks]([])
    ax_x[:set_yticks]([])

    ax_y = fig[:add_axes]([0.62, 0.24, 0.08, 0.60])
    ax_y[:plot](sum(W, dims=2)[:] .* (xs[2]-xs[1]), ys)  # marginal along y
    ax_y[:set_xticks]([])
    ax_y[:set_yticks]([])

    tight_layout()
    return fig
end



function get_collective_samples(sim; time_idx::Union{Int,Nothing}=nothing,
                                weights=nothing, mode::Symbol=:linear,
                                normalize_weights::Bool=true,
                                circular_positions::Bool=false)
    sols = try
        extract_solution(sim)
    catch
        sim
    end
    if isempty(sols)
        error("No trajectories in sim")
    end
    Nt = length(sols[1].t)
    t_idx = time_idx === nothing ? Nt : time_idx
    Ntrajs = length(sols)
    Nsys = sols[1].p.N

    samples = zeros(Float64, Ntrajs, 2)  # (X_coll, P_coll) rows

    # Pre-handle case when weights is fixed vector
    fixed_weights = nothing
    if weights isa AbstractVector
        if length(weights) != Nsys
            error("weights vector length must equal N (got $(length(weights)) vs N=$Nsys)")
        end
        fixed_weights = copy(weights)
        if normalize_weights
            s = sum(abs.(fixed_weights))
            s == 0.0 && (s = 1.0)
            fixed_weights ./= s
        end
    end

    for (i, sol) in enumerate(sols)
        if length(sol.t) < t_idx
            error("Trajectory $(i) does not have time index $t_idx")
        end
        x = sol.u[1:Nsys, t_idx]        # positions (may be angles)
        p = sol.u[Nsys+1:2Nsys, t_idx]  # momenta

        # choose weights for this trajectory
        w = fixed_weights === nothing ? ones(Nsys) : fixed_weights

        if weights === :cospos
            w = cos.(x)
        elseif weights === :sinpos
            w = sin.(x)
        elseif weights === nothing && fixed_weights === nothing
            w .= 1.0
        elseif weights isa AbstractVector
            # fixed_weights already handled
        else
            # Unknown symbol/option -> error to avoid silent mistakes
            if fixed_weights === nothing
                error("Unknown `weights` option; pass a vector of length N or :cospos / :sinpos or nothing")
            end
        end

        # Normalize weights if requested and not already normalized
        if normalize_weights && fixed_weights === nothing
            s = sum(abs.(w))
            s == 0.0 && (s = 1.0)
            w = w ./ s
        end

        # Compute collective X and P
        if circular_positions
            # compute circular mean for positions (angle mean) then use it as X_coll
            sx = mean(sin.(x) .* w) 
            cx = mean(cos.(x) .* w)
            Xc = atan(sx, cx)        


            Pc = dot(w, p)
        else
            Xc = dot(w, x)
            Pc = dot(w, p)
        end

        samples[i, 1] = Xc
        samples[i, 2] = Pc
    end

    return samples
end


function plot_collective_wigner(sim; time_idx=nothing, weights=nothing, normalize_weights=true,
                                circular_positions=false, nbins::Int=201, hscale::Float64=1.0,
                                cmap="RdBu_r", show_contours=false)

    samples = get_collective_samples(sim; time_idx=time_idx, weights=weights,
                                     normalize_weights=normalize_weights,
                                     circular_positions=circular_positions)


    xs, ys, W = kde2d(samples; nbins=nbins, hscale=hscale)

    fig, ax = subplots(figsize=(6,4.5))
    extent = (first(xs), last(xs), first(ys), last(ys))
    im = ax[:imshow](W, origin="lower", extent=extent, aspect="auto", interpolation="bilinear",cmap="inferno")#,vmin=-maximum(abs.(W)), vmax=maximum(abs.(W)),cmap="bone")
    ax[:set_xlabel](L"X_\mathrm{coll}"); ax[:set_ylabel](L"P_\mathrm{coll}")
    cbar = fig[:colorbar](im, ax=ax)
    cbar[:set_label](L"W(X_\mathrm{coll},P_\mathrm{coll})")

    if show_contours
        levels = range(minimum(W), stop=maximum(W), length=7)[2:end-1]
        ax[:contour](xs, ys, W, levels=collect(levels), linewidths=1.0)
    end

    # marginal axes
    ax_x = fig[:add_axes]([0.2, 0.85, 0.50, 0.08])
    ax_x[:plot](xs, sum(W, dims=1)[:] .* (ys[2]-ys[1]))
    ax_x[:set_xticks]([]); ax_x[:set_yticks]([])

    ax_y = fig[:add_axes]([0.62, 0.24, 0.08, 0.60])
    ax_y[:plot](sum(W, dims=2)[:] .* (xs[2]-xs[1]), ys)
    ax_y[:set_xticks]([]); ax_y[:set_yticks]([])

    tight_layout()
    return fig
end





#########################################################
#           Emission statistics                         #
#########################################################

function plot_g2(sim::Array{Sol,1})
    # === Compute expectations and form g2(t) ===
    mean_adaga, std_adaga, q90_adaga = expect(obs_adaga, sim)
    mean_adaga2, std_adaga2, q90_adaga2 = expect(obs_adaga2, sim)
    t = (extract_solution(sim))[1].t  
    # numerator = <n^2> - <n>
    numer = mean_adaga2 .- mean_adaga;
    # guard against tiny denominators
    eps_small = 1e-12;
    denom = mean_adaga .^ 2 .+ eps_small;
    g2_mean = numer ./ denom;
    g2_mean[mean_adaga .< 1e-8] .= NaN;
    
    fig, ax = subplots(figsize=(8,3))
    ax.plot(t, g2_mean, lw=2, label=L"g^{(2)}(t)")
    xlabel(L"t"); ylabel(L"g^{(2)}(t)")
    ylim(0.5,2.5)
    tight_layout()

    return fig
end