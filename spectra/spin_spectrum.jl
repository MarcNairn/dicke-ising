using Base.Threads
Threads.nthreads()  

using DelimitedFiles
using Accessors

include("../src/main.jl")
include("../src/plotting.jl")
include("../src/macros.jl")


function initial_conditions(p::System_p, rng::AbstractRNG)
    N::Int = p.N
    u0 = zeros(5N + 2)


    u0[1:N] .= 2π .* rand(rng,N)
    u0[N+1:2N] .= p.temp .* randn(rng, N)
    u0[2N+1:4N] .= 2 .* (rand(rng, 2N) .< 0.5) .- 1
    u0[4N+1:5N] .= -1.0 #spin down


    u0[5N+1:end] .= 0.0

    return u0
end


# --- Updated define_prob_from_parameters: create per-trajectory copies of p and per-trajectory RNG ---
function define_prob_from_parameters(p::System_p; seed::Int = abs(rand(Int)))

    rng0 = MersenneTwister(seed)
    u0_example = initial_conditions(p, rng0)

    prob = SDEProblem(f_det, f_noise, u0_example, p.tspan, p)

    function prob_func(prob, i, repeat)
        p_copy = deepcopy(p)
        rng = MersenneTwister(seed + i)
        u0_traj = initial_conditions(p_copy, rng)

        # RETURN A NEW PROBLEM (do NOT mutate prob)
        return SDEProblem(f_det, f_noise, u0_traj, p.tspan, p_copy)
    end

    monte_prob = EnsembleProblem(prob; prob_func = prob_func)

    return prob, monte_prob
end

# --- Updated many_trajectory_solver supporting threads   ---
function many_trajectory_solver(
        p::System_p;
        saveat::Real = 10.0,
        seed::Int = abs(rand(Int)),
        backend::Symbol = :threads,   # :threads or :distributed
        alg = SOSRA2(),
        trajectories::Int = p.N_MC,
        maxiters::Int = Int(1e9)
    )

    prob, monte_prob = define_prob_from_parameters(p; seed=seed)

    # select ensemble backend
    ensemble_alg =
        backend == :threads ? EnsembleThreads() :
        backend == :distributed ? EnsembleDistributed() :
        error("backend must be :threads or :distributed")

    println("Starting ensemble solve with backend = $backend, trajectories = $trajectories ...")
    elt = @elapsed sim = solve(monte_prob, alg, ensemble_alg;
                               trajectories = trajectories,
                               saveat = saveat,
                               maxiters = maxiters,
                               progress = true)

    println("done in $elt seconds.")
    return sim
end

#####

# Conversion from collective coupling Ω to single particle g (ωR units): 
#  g = √[(-κ²+Δc²/(2√2NΔc)⋅1/∫₀^∞ dt e^(-(2t)²tc²/2)sin(ω₀t))]

####
function gc(ω₀, Δc, κ, temp, N)
    fun(t) = exp(-(2*temp)^2 * t^2 / 2) * (sin(ω₀ * t))
    Intg, _ = quadgk(fun, 0, Inf; order=9)
    return (κ^2 + Δc^2)*1/(-2√2*N*Δc)*1/Intg
end

Δ = 100; ω₀ = 0.10Δ; κ = Δ; Δc = -Δ; tspan = (0,2.0)
N = 50; N_MC = 50;temp=0.05Δ; rthresh = 2pi/5.3;V= 2.0*g0;
dressed_idx = collect(1:N);

V_new =  0.0;
t_quench = Inf;

gbase  = gc(ω₀, Δc, κ, temp, N); #Threshold as base unit for single atom g
g0 = 2*gbase#2.0 * gbase;

p = System_p(g0, ω₀, Δc, κ, temp, N, tspan, N_MC, V, rthresh, dressed_idx;
             t_quench = t_quench,
             V_after  = V_new);




Vrange = range(0.0, 2*g0, 15);

grange = range(0,1.0*gbase,20);

# --- Storage arrays ---
Skzs = []
Skxs = []
qmax = 4.0  
dq   = 0.1  #  q-resolution
qs   = collect(-qmax:dq:qmax)

function run_spectrum(modified_p::System_p; seed_offset::Int=0, integrated=false,η=0.5)
    # Change parameter of choice in struct
    sim = many_trajectory_solver(
        modified_p;
        saveat       = 0.1,
        seed         = 1234 + seed_offset,
        backend      = :threads,
        trajectories = modified_p.N_MC
    )
    qz_I = Float64[]
    qx_I = Float64[]
    for q in qs
        spectrumz, _, _ = expect(DSSF_Z_fft(q), sim)
        spectrumz = fftshift(spectrumz)

        spectrumx, _, _ = expect(DSSF_X_fft(q), sim)
        spectrumx = fftshift(spectrumx)

        N = length(spectrumz)
        dt = 0.1
        freqs = fftshift(fftfreq(N, 1/dt))
        # "Punishment" kernel
        kernel = 1 ./ (freqs.^2 .+ η^2)
        
        # Identify the index for omega = 0 (midpoint of signal, after fft)
        # Should also coincide with MAX(spectrum)
        idz = Int(floor(length(spectrumz) / 2)) + 1
        idx = Int(floor(length(spectrumx) / 2)) + 1

        if integrated==true
            # If decay is present in signal, better to evaluate via weighted frequency kernel

            weighted_z = sum(abs.(spectrumz) .* kernel) 
            weighted_x = sum(abs.(spectrumx) .* kernel)

            push!(qz_I, weighted_z)
            push!(qx_I, weighted_x)
        else
            push!(qz_I, spectrumz[idz])
            push!(qx_I, spectrumx[idx])
        end
    end

    return qz_I, qx_I
end


Rrange = range(0,2pi, 10)


for (ri,r_val) in enumerate(Rrange)
    p_current = @set p.rthresh = r_val # Accesors.jl helper @set to dynamically change struct entries

    spectrumz, spectrumx = run_spectrum(p_current, integrated=true, η=0.1)
    push!(Skzs, spectrumz)
    push!(Skxs, spectrumx)
    total = length(Rrange)
    print("Finished run $ri/$total")

end




writedlm("Spectra_sz.txt", Skzs)
writedlm("Spectra_sx.txt", Skxs)






# vs g 

Zx = hcat(Skxs...)
Zz = hcat(Skzs...)
Zx_norm = Zx ./ maximum(Zx)
Zz_norm = Zz ./ maximum(Zz)

fig, axs = subplots(2, 1, figsize=(4, 3.5), sharex=true)
ax1, ax2 = axs
# fig.subplots_adjust(wspace=0.0)
fig.subplots_adjust(hspace=0.0)
pm1 = ax1.pcolormesh(grange/gbase, qs, Zx_norm, cmap="binary", shading="gouraud")
ax1.text(0.05, 0.9, L"S_x(q)", transform=ax1.transAxes, fontweight="bold", va="top")
ax1.set_ylabel(L"q/k_c")
ax1.set_yticks([-2, 0, 2])
ax1.axhline(-1, color="grey", ls="--", lw=2, alpha=0.5)
ax1.axhline(1, color="grey", ls="--", lw=2,alpha=0.5)
ax1.tick_params(direction="in", which="both", top=true, right=true)


pm2 = ax2.pcolormesh(grange/gbase, qs, Zz_norm, cmap="binary", shading="gouraud")
ax2.text(0.05, 0.9, L"S_z(q)", transform=ax2.transAxes, fontweight="bold", va="top")
ax2.set_ylabel(L"q/k_c")
ax2.set_xlabel(L"g/g_c")
ax2.set_yticks([-2, 0,2]) 
ax2.tick_params(direction="in", which="both", top=true, right=true)

cax = fig.add_axes([0.89, 0.34, 0.04, 0.40])  #[left, bottom, width, height]

cb = fig.colorbar(pm2, cax=cax)
cb.set_label(L"\mathcal{I}_\mathrm{eff}(q)", labelpad=-10)
cb.outline.set_linewidth(1.25)
cb.ax.tick_params(width=1.25,direction="in")
cb.set_ticks([0, 1.0])
fig.subplots_adjust(hspace=0.0, left=0.15, right=0.85, bottom=0.15, top=0.95)

#tight_layout()
display(fig)




# vs V

Zx = readdlm("Spectra_sx.txt", Float64)
Zz = readdlm("Spectra_sz.txt", Float64)

Zx_norm = Zx ./ maximum(Zx)
Zz_norm = Zz ./ maximum(Zz)

fig, axs = subplots(2, 1, figsize=(4, 3.5), sharex=true)
ax1, ax2 = axs
# fig.subplots_adjust(wspace=0.0)
fig.subplots_adjust(hspace=0.0)
pm1 = ax1.pcolormesh(Vrange/g0, qs, Zx_norm, cmap="RdGy_r", shading="gouraud", norm=LogNorm(vmin=5e-2, vmax=1e0))
ax1.text(0.05, 0.9, L"S_x(q)", transform=ax1.transAxes, fontweight="bold", va="top", color="white")
ax1.set_ylabel(L"q/k_c")
ax1.set_yticks([-2, 0, 2])
ax1.axhline(-1, color="grey", ls="--", lw=2, alpha=0.5)
ax1.axhline(1, color="grey", ls="--", lw=2,alpha=0.5)
ax1.tick_params(direction="in", which="both", top=true, right=true)


pm2 = ax2.pcolormesh(Vrange/g0, qs, Zz_norm, cmap="RdGy_r", shading="gouraud", norm=LogNorm(vmin=5e-2, vmax=1e0))
ax2.text(0.05, 0.93, L"S_z(q)", transform=ax2.transAxes, fontweight="bold", va="top",color="white")
ax2.set_ylabel(L"q/k_c")
ax2.set_xlabel(L"V/g")
ax2.set_yticks([-2, 0,2]) 
ax2.tick_params(direction="in", which="both", top=true, right=true)
# ax2.axhline(-2, color="grey", ls="-.", lw=2, alpha=0.5)
# ax2.axhline(2, color="grey", ls="-.", lw=2,alpha=0.5)
cax = fig.add_axes([0.89, 0.34, 0.04, 0.40])  #[left, bottom, width, height]

cb = fig.colorbar(pm2, cax=cax)
cb.set_label(L"\mathcal{I}_\mathrm{eff}(q)")
cb.outline.set_linewidth(1.25)
cb.ax.tick_params(width=1.25,direction="in")
cb.set_ticks([1e-1, 1.0])
cb.set_ticklabels([0.1, 1])
fig.subplots_adjust(hspace=0.0, left=0.15, right=0.85, bottom=0.15, top=0.95)

#tight_layout()
display(fig)








fig, axs = subplots(2, 1, figsize=(4, 3.5), sharex=true)
ax1, ax2 = axs
# fig.subplots_adjust(wspace=0.0)
fig.subplots_adjust(hspace=0.0)
pm1 = ax1.pcolormesh(Rrange, qs, Zx_norm, cmap="RdGy_r", shading="gouraud", norm=LogNorm(vmin=5e-2, vmax=1e0))
ax1.text(0.05, 0.9, L"S_x(q)", transform=ax1.transAxes, fontweight="bold", va="top", color="white")
ax1.set_ylabel(L"q/k_c")
ax1.set_yticks([-2, 0, 2])
# ax1.axhline(-1, color="grey", ls="--", lw=2, alpha=0.5)
# ax1.axhline(1, color="grey", ls="--", lw=2,alpha=0.5)
ax1.tick_params(direction="in", which="both", top=true, right=true)


pm2 = ax2.pcolormesh(Rrange, qs, Zz_norm, cmap="RdGy_r", shading="gouraud", norm=LogNorm(vmin=5e-2, vmax=1e0))
ax2.text(0.05, 0.93, L"S_z(q)", transform=ax2.transAxes, fontweight="bold", va="top",color="white")
ax2.set_ylabel(L"q/k_c")
ax2.set_xlabel(L"R_c/\lambda_c")
ax2.set_xticks([0, pi, 2pi],[L"0", "0.5", "1"])
ax2.set_yticks([-2, 0,2]) 
ax2.tick_params(direction="in", which="both", top=true, right=true)
# ax2.axhline(-2, color="grey", ls="-.", lw=2, alpha=0.5)
# ax2.axhline(2, color="grey", ls="-.", lw=2,alpha=0.5)
cax = fig.add_axes([0.89, 0.34, 0.04, 0.40])  #[left, bottom, width, height]

cb = fig.colorbar(pm2, cax=cax)
cb.set_label(L"\mathcal{I}_\mathrm{eff}(q)")
cb.outline.set_linewidth(1.25)
cb.ax.tick_params(width=1.25,direction="in")
cb.set_ticks([1e-1, 1.0])
cb.set_ticklabels([0.1, 1])
fig.subplots_adjust(hspace=0.0, left=0.15, right=0.85, bottom=0.15, top=0.95)

#tight_layout()
display(fig)