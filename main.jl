
using Base.Threads
Threads.nthreads()  

include("src/main.jl")
include("src/plotting.jl")
include("src/macros.jl")





###
# Main change in multithreading: fixed rng at each sample
###

function initial_conditions(p::System_p, rng::AbstractRNG)
    N::Int = p.N
    u0 = zeros(5N + 2)
    # Uniform spacing
    # for j in 1:N
    #     u0[j] = 2pi*(j-1)/N
    # end

    # Place at cavity nodes to minimize cavity coupling
    # for j in 1:div(N,2)
    #     u0[j] = pi/2 + 0.01*randn(rng)
    # end
    # for j in div(N,2)+1:N
    #     u0[j] = 3pi/2 + 0.01*randn(rng)
    # end

    u0[1:N] .= 2π .* rand(rng,N)
    u0[N+1:2N] .= p.temp .* randn(rng, N)
    u0[2N+1:4N] .= 2 .* (rand(rng, 2N) .< 0.5) .- 1
    u0[4N+1:5N] .= -1.0 #spin down

    # Domain wall σz
    # for j in 1:N
    #     u0[4N+j] = j <= N/2 ? 1.0 : -1.0
    # end
    
    # cavity empty
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


function traj_Z_double(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [mean(sol.u[4N+1:5N,j].*cos.(2* sol.u[1:N,j])) for j=1:length(sol.t)]
end
Zdouble = Observable(traj_Z_double,L"N^{-1}\sum_j\sigma_z\cos(x_j)",L"Z")


Δ = 100 # Relative frequency unit so that we work in dispersive regime ωR << Δ
# g = 0.1*Δ
ω₀ = 0.3Δ
Δc = -1*Δ
κ = 1*Δ
temp = 0.05*Δ
N = 50

# Conversion from collective coupling Ω to single particle g (ωR units): g = √[(-κ²+Δc²/(2√2NΔc)⋅1/∫₀^∞ dt e^(-(2t)²tc²/2)sin(ω₀t))]
function gc(ω₀, Δc, κ, temp, N)
    fun(t) = exp(-(2*temp)^2 * t^2 / 2) * (sin(ω₀ * t))
    Intg, _ = quadgk(fun, 0, Inf; order=9)
    return (κ^2 + Δc^2)*1/(-2√2*N*Δc)*1/Intg
end

g = 2.0*gc(ω₀, Δc, κ, temp, N)

tspan = (0,1.0)
N_MC = 50 
rthresh = 2pi/8 # as fraction of λc -> k_b = 2π/(λc*rthresh)
V =0*g
dressed_idx = collect(1:N)

# Quench parameters
V_new =  0#1.0*g
t_quench = Inf#tspan[end]/2

p = System_p(g, ω₀, Δc, κ, temp, N, tspan, N_MC, V, rthresh, dressed_idx;
             t_quench = t_quench,
             V_after  = V_new);


sim = many_trajectory_solver(p;saveat = 0.01,seed = 1234,backend = :threads,trajectories = p.N_MC)

t = sim[1].t

plus, minus, _ = categorize_traj(extract_solution(sim));
plot_spinspositionhisto(extract_solution(minus));
plot_spinphasehisto(extract_solution(sim));

mean_adaga, std_adaga, q90_adaga = expect(adaga, sim);
q90_adaga = hcat(q90_adaga...);

Xp, _, q90_Xp = expect(X, plus);
q90_Xp = hcat(q90_Xp...);

Xm, _, q90_Xm = expect(X, minus);
q90_Xm = hcat(q90_Xm...);

Z2, _, q90_Z = expect(Zdouble, sim);
q90_Z = hcat(q90_Z...);

Jz, _ , q90_Jz = expect(Sz, sim);
q90_Jz = hcat(q90_Jz...);

fig, ax = subplots(figsize=(4,3))
ax.plot(t, Jz, lw=2, color="teal")
display(fig)

mean_phase, std_phase, q90_phase = expect(a_phase, sim)
q90_phase = hcat(q90_phase...)



fig, ax = subplots(figsize=(4,3))
ax.plot(t, Xm, lw=2, color="crimson")
ax.fill_between(t, q90_Xm[1,:], q90_Xm[2,:], alpha=0.1, color="crimson")
ax.plot(t, Xp, lw=2, color="dodgerblue")
ax.fill_between(t, q90_Xp[1,:], q90_Xp[2,:], alpha=0.1, color="dodgerblue")
ax.set_ylim(-1,1)
display(fig)




fig, ax = subplots(figsize=(4,3))
ax.plot(t, mean_adaga, lw=2, color="black")
ax.fill_between(t, q90_adaga[1,:], q90_adaga[2,:], alpha=0.1, color="grey")

ax.plot(t, Z2, lw=2, color="teal")
ax.fill_between(t, q90_Z[1,:], q90_Z[2,:], alpha=0.1, color="silver")
ax.set_xlabel(L"gt")
ax.set_ylabel(L"Z")
#ax.axvline(t_quench, lw=2, c="grey")
display(fig)


mean_ar, std_ar, q90_ar = expect(ar, sim)
mean_ai, std_ai, q90_ai = expect(ai, sim)

q90_ar = hcat(q90_ar...)
q90_ai = hcat(q90_ai...)

mean_a = mean_ar .+ im .* mean_ai


fig, ax = subplots(figsize=(4,3))
ax.plot(t, mean_phase, lw=2, color="black")
ax.fill_between(t, q90_phase[1,:], q90_phase[2,:], alpha=0.1, color="grey")
ax.set_xlabel(L"gt")
ax.set_ylabel(L"\mathrm{arg}(a)")
ax.axvline(t_quench, lw=2, c="grey",ls="--")
ax.set_ylim(-2,2)
ax.set_yticks([-pi/2,-pi/4,0, pi/4,pi/2], [L"-\pi/2", L"-\pi/2", 0, L"\pi/4", L"\pi/2"])
display(fig)








# S² =  ∑α ∑ⱼ (σⱼ^α)^2  /N
function traj_spinlength(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [mean(sol.u[2N+1:3N,j].^2 + sol.u[3N+1:4N,j].^2 + sol.u[4N+1:5N,j].^2) for j=1:length(sol.t)]
end
S2 = Observable(traj_spinlength,L"N^{-1}\sum_j\sigma_\alpha",L"Slen")

S2len, _, _ = expect(S2, sim);


# φ = arctan(σⱼᶻ/σⱼˣ)*cos(k_cxⱼ) 

function traj_spinphase(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    x = view(sol.u, 1:N, :)          
    σx = view(sol.u, 2N+1:3N, :)     
    σz = view(sol.u, 4N+1:5N, :)     
    [mean((atan.(σz[:, j] ./ σx[:, j]) .* cos.( x[:, j]))) for j = 1:length(sol.t)]
end

S_phase = Observable(
    traj_spinphase,
    L"N^{-1}\sum_j \arctan(\sigma_z/\sigma_x) \cos(k_c x_j)",
    L"\Phi_{\mathrm{cos}}"
)
plus, minus, _ = categorize_traj(extract_solution(sim));


sphase_p, _, q90_phase_p = expect(S_phase, plus);
sphase_m, _, q90_phase_m = expect(S_phase, minus);
sphase, _, q90_phase = expect(S_phase, sim);

q90_phase_p = hcat(q90_phase_p...)
q90_phase_m = hcat(q90_phase_m...);
q90_phase = hcat(q90_phase...);

fig, ax = subplots(figsize=(4,3))
ax.plot(t, sphase_p, lw=2, color="crimson")
ax.plot(t, sphase_m, lw=2, color="dodgerblue")
ax.plot(t, sphase, lw=2, color="black")
ax.fill_between(t, q90_phase_p[1,:], q90_phase_p[2,:], alpha=0.1, color="grey")
ax.fill_between(t, q90_phase_m[1,:], q90_phase_m[2,:], alpha=0.1, color="lightblue")
ax.set_xlabel(L"gt")
ax.set_ylabel(L"\Phi_S")
ax.axvline(t_quench, lw=2, c="grey", ls="--")
ax.set_ylim(-0.2,0.2)
display(fig)




function traj_pair_broadening(sol::Sol)
    N = try sol.p.N catch; sol.p[10] end
    nt = length(sol.t)

    L = 2π
    halfL = π

    B = zeros(nt)

    for j in 1:nt
        x = sol.u[1:N, j]
        s = 0.0

        for i in 1:N-1
            for k in i+1:N
                Δ = x[i] - x[k]

                # wrap to [-π, π]
                Δ = mod(Δ + halfL, L) - halfL

                s += Δ^2
            end
        end

        B[j] = 2s / (N*(N-1))
    end

    return B
end

PairBroadening = Observable(
    traj_pair_broadening,
    L"\frac{1}{N(N-1)}\sum_{i\ne j}\mathrm{wrap}(x_i-x_j)^2",
    L"B_{\mathrm{pair}}",
    "density-induced motional broadening"
)


mean_B, std_B, q90_B = expect(PairBroadening, sim);
q90_B = hcat(q90_B...);



fig, ax = subplots(figsize=(3,2));
ax.plot(t, mean_B, lw=2,color="black");
ax.fill_between(t, q90_B[1,:], q90_B[2,:], alpha=0.1, color="grey");
ax.set_ylabel(L"\beta_x");
ax.set_xlabel(L"gt",labelpad=-15);
ax.axhline(pi^2/3, lw=2,color="grey", label="uni");
ax.axvline(t_quench, lw=2, color="teal", ls="--");
ax.legend()
display(fig)

function traj_fano(sol::Sol; M::Int=10)
    N = try sol.p.N catch; sol.p[10] end
    nt = length(sol.t)
    F = zeros(nt)

    nbar = N / M
    var_max = N^2 * (M-1) / M^2

    for j in 1:nt
        x = sol.u[1:N, j]  # wrapped positions [0, 2π)
        n_alpha = zeros(M)

        for xi in x
            # compute bin index
            idx = Int(floor(M * xi / (2π))) + 1
            idx = min(max(idx, 1), M)  # clamp
            n_alpha[idx] += 1
        end

        var_n = var(n_alpha)

        F[j] = var_max > 0 ? var_n / var_max : 0.0
    end

    return F
end

# Wrap it as an Observable
Fano = Observable(
    traj_fano,
    L"F = \mathrm{Var}(n_\alpha)/\bar n",
    L"F_\mathrm{radial}",
    "radial Fano factor (local density fluctuations)"
)

mean_F, std_F, q90_F = expect(Fano, sim);
q90_F = hcat(q90_F...);





#######################################

"""
Phase diagram of magnetization and spin texture
"""
gbase  = gc(ω₀, Δc, κ, temp, N); #Threshold as base unit for single atom g

g0 = 2.5 * gbase
V0 = 1.25 * gbase

base_params = System_p(g, ω₀, Δc, κ, temp, N, tspan, N_MC, V, rthresh, dressed_idx;
             t_quench = t_quench,
             V_after  = V_new);

Rcs= range(2pi/20, 2pi, length=21);

# --- Storage arrays ---
Jz_mean  = Float64[]
Jz_std   = Float64[]
X2_mean  = Float64[]
X2_std   = Float64[]
Phi_mean = Float64[]
Phi_std  = Float64[]

function run_one_rthresh(rthresh_new::Float64, base_p::System_p; seed_offset::Int=0)
    p_scan = System_p(
        g0, base_p.ω₀, base_p.Δc, base_p.κ,
        base_p.temp, base_p.N, base_p.tspan, base_p.N_MC,
        V0, rthresh_new, base_p.dressed_idx;
        t_quench = Inf,   # no quench
        V_after  = 0
    )

    sim = many_trajectory_solver(
        p_scan;
        saveat       = 0.10,
        seed         = 1234 + seed_offset,
        backend      = :threads,
        trajectories = p_scan.N_MC
    )

    plus, _, _ = categorize_traj(extract_solution(sim))

    if length(plus) > 0.1 * p_scan.N_MC   # symmetry broken?
        sols = extract_solution(plus)
    else
        sols = extract_solution(sim)
    end

    jz_mean, jz_std, _ = expect(Sz, sols)
    X2_mean, X2_std, _ = expect(X2, sim)
    phi_mean, phi_std, _ = expect(S_phase, sols)

    return (jz_mean[end], jz_std[end],
            X2_mean[end], X2_std[end], phi_mean[end], phi_std[end])
end

global_counter = 0
base_p = p
for (idx, rth) in enumerate(Rcs)
    global_counter += 1
    println("r_thresh = $(round(rth, digits=3))  ($idx/$(length(Rcs)))")

    jz_m, jz_s, X2_m, X2_s, phi_m, phi_s =
        run_one_rthresh(rth, base_p; seed_offset=global_counter)

    push!(Jz_mean,  jz_m)
    push!(Jz_std,   jz_s)
    push!(X2_mean,  X2_m)
    push!(X2_std,   X2_s)
    push!(Phi_mean, phi_m)
    push!(Phi_std,  phi_s)

    GC.gc() #Free memory (NEED WITH THREADS() )
end


fig, ax = plt.subplots(figsize=(5,4))

# ax.errorbar(Rcs./ 2pi, Jz_mean, yerr=Jz_std,
#             fmt="o-", capsize=3, color="teal", ecolor="lightgray",
#             label=L"\langle J_z \rangle")

# ax.errorbar(Rcs./ 2pi, Phi_mean, yerr=Phi_std,
#             fmt="s-", capsize=3, color="purple", ecolor="lightgray",
#             label=L"\Phi_S")

ax.errorbar(Rcs./ 2pi, X2_mean, yerr=X2_std,
            fmt="s-", capsize=3, color="coral", ecolor="lightgray",
            label=L"\langle X\rangle^2")

ax.set_xlabel(L"k_c \cdot R_b")
ax.set_ylabel(L"\langle O \rangle")
ax.legend()
display(fig)


# --- Plot ---
fig, ax = subplots(figsize=(5,4))
ax.errorbar(Vs/g, Jz_mean, yerr=Jz_std, fmt="o-", capsize=3,color="teal", ecolor="lightgray", label=L"\langle J_z\rangle")
ax.errorbar(Vs/g, Phi_mean, yerr=Phi_std, fmt="o-", capsize=3,color="purple", ecolor="lightgray", label=L"\Phi_S")
ax.set_xlabel(L"V / g")
ax.set_ylabel(L"\langle O \rangle")
#ax.grid(alpha=0.3)
ax.legend()
display(fig)



qs = range(-π, π, length=101)   
nt = length(sim[1].t)
dt = sim[1].t[2] - sim[1].t[1]
T = nt * dt

# Frequency grid
ω = 2π .* fftfreq(nt, 1/dt)
ω = fftshift(ω)   # center zero frequency

Sqω_list = []
for q in qs
    Sqw,_,_ = expect(DSSF_X_fft(q), sim)  # mean
    push!(Sqω_list, Sqw)
end
Sqω = hcat(Sqω_list...);        
Sqω = fftshift(Sqω, 1);         



fig, ax = subplots(figsize=(4,3))
imshow(Sqω*1/maximum(Sqω),aspect="auto",extent=(minimum(qs), maximum(qs), minimum(ω), maximum(ω)),origin="lower",cmap="afmhot", norm=LogNorm(vmin=1e-2,vmax=1))
colorbar(label=L"S^{xx}(q,\omega)")
xlabel(L"q / k_c")
ylabel(L"\omega/\omega_R")
ylim(-10,10)
#axvline(1, label=L"k_c", color="white", ls="--")
#axvline(rthresh, label=L"R_c", color="purple")
axvline(1.0, c="white",  label=L"\pm k_c")
axvline(-1.0, c="white")
axvline(2.0, c="white", label = L"\pm 2k_c", ls="--")
axvline(-2.0, c="white", ls="--")
legend(fontsize=12, loc="upper center")
# tight_layout()
display(fig)


##########

# TO FIX #

# TRUE Spin structure factor as a correlator



function traj_autocorr_Z(sol::Sol, q::Float64)
    N = sol.p.N

    Sq_t = [mean(sol.u[4N+1:5N,j] .* exp.(im*q .* sol.u[1:N,j])) for j=1:length(sol.t)]
    
    correlation = [mean(Sq_t[τ:end] .* conj(Sq_t[1:end-τ+1])) for τ in 1:length(Sq_t)]
    return correlation
end


function corr_DSSF_Z(sols::Vector{Sol}, q::Float64)
    # Average the correlations across different realizations (trajectories)
    avg_corr = mean([traj_autocorr_Z(s, q) for s in sols])

    return real.(fft(avg_corr))
end


Sqnew = []
for q in qs
    Sq= corr_DSSF_Z(extract_solution(sim), q)
    push!(Sqnew, Sq)
end


Sqnew = hcat(Sqnew...);
Sqnew = fftshift(Sqnew, 1);     

##########

