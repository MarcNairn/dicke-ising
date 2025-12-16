include("src/main.jl")
include("src/plotting.jl")

#using PyCall

PyPlot.matplotlib[:rc]("text", usetex=true)
PyPlot.matplotlib[:rc]("font", family="serif", size = 18)#serif=["mathpazo"], size=18)  # Base font size
PyPlot.matplotlib[:rc]("axes", titlesize=30)             # Axis title
PyPlot.matplotlib[:rc]("axes", labelsize=30)             # Axis labels
PyPlot.matplotlib[:rc]("xtick", labelsize=24)            # X-ticks
PyPlot.matplotlib[:rc]("ytick", labelsize=24)            # Y-ticks
PyPlot.matplotlib[:rc]("legend", fontsize=24)            # Legend
PyPlot.matplotlib[:rc]("figure", titlesize=24)           # Figure title
PyPlot.svg(true)
# LaTeX preamble packages
PyPlot.matplotlib[:rc]("text.latex", preamble="\\usepackage{amsmath}\\usepackage{amsfonts}\\usepackage{amssymb}\\usepackage{lmodern}")


g = 20
ω₀ = 20.0
Δc = -100.0
κ = 100.0
temp = 10.0
N = 100
tspan = (0,2)
N_MC = 30
# Rydberg/dressed params 
#fR = 0.1 # fraction of dressed atoms (0.0..1.0)
V = -5 # Ising strength between dressed atoms
rthresh = 0.01 # cutoff distance for chi (in same units as x mod 2π)

sqrt.(-1 * (κ^2 + Δc^2) / (2 * Δc) / N) * sqrt.(g)




dressed_idx = randperm(N)[1:N]; # Does nothing for now

p = System_p(g, ω₀, Δc, κ, temp, N, tspan, N_MC, V, rthresh, dressed_idx)


sim = many_trajectory_solver(p, saveat=0.01)


mean_X, std_X, q90_X = expect(obs_X2, sim)
q90_X = hcat(q90_X...)
mean_adaga, std_adaga, q90_adaga = expect(obs_adaga, sim)
q90_adaga = hcat(q90_adaga...)
mean_Z, std_Z, q90_Z = expect(obs_Jz, sim)
q90_Z = hcat(q90_Z...)
t = sim[1].t


fig, ax = subplots(figsize=(8,3))
plot(t, mean_adaga , lw=2)
fill_between(t, q90_adaga[1,:],q90_adaga[2,:], alpha=0.25)
xlabel(L"t/\omega_R"); ylabel(L"\langle a^\dagger a \rangle");
display(fig)


fig, ax = subplots(figsize=(8,3))   
plot(t, mean_Z , lw=2, color="black")
fill_between(t, q90_Z[1,:],q90_Z[2,:], alpha=0.25, color="grey")
xlabel(L"t/\omega_R"); ylabel(L"\langle J_z\rangle");
ylim(-1.05, 1.05);
display(fig)




fig = plot_contact_count(sim)



fig, _ = plot_position(extract_solution(sim))
display(fig)


plus, minus, other = categorize_traj(extract_solution(sim));


fig, _ = plot_position(extract_solution(sim), length(sim[1].t))
fig, _ = plot_spinspositionhisto(extract_solution(minus))






plot_contact_spatial_correlation(sim)


fig = plot_cavity_wigner(sim,show_contours=false)

fig = plot_collective_wigner(sim)








#############################################################

                    #Phase transition#

#############################################################


Vs = range(-2.5, 5, length=10);

end_mean_adaga = zeros(length(Vs));
end_q90_adaga = Array{Float64}(undef, 2, length(Vs));
end_mean_X2 = zeros(length(Vs));
end_q90_X2 = Array{Float64}(undef, 2, length(Vs));

for (k, V) in enumerate(Vs)

    println("Running V = $V")

    dressed_idx = randperm(N)[1:N] 

    p = System_p(g, ω₀, Δc, κ, temp, N, tspan, N_MC, V, rthresh, dressed_idx)

    sim = many_trajectory_solver(p, saveat=0.02)

    mean_adaga, std_adaga, q90_adaga = expect(obs_adaga, sim)

    end_mean_adaga[k] = mean_adaga[end]   

    end_q90_adaga[:, k] = q90_adaga[end]

    mean_X2, _, q90_X2 = expect(obs_X2, sim)

    end_mean_X2[k] = mean_X2[end]
    end_q90_X2[:, k] = q90_X2[end]
end

fig, ax = subplots(figsize=(6,3))
ax.plot(Vs, end_mean_adaga, "black", lw=3)
ax.fill_between(Vs, end_q90_adaga[1, :], end_q90_adaga[2, :], color="grey", alpha=0.25)
ax.set_xlabel(L"V/g")
ax.set_xticks([-2.5,0,2.5,5],[L"-0.25",L"0",L"0.25",L"0.5"])
#ax.set_ylabel(L"\langle\hat{n}_\mathrm{cav}(t_f)\rangle")
ax.set_ylabel(L"\langle\hat{X}^2(t_f)\rangle")
tight_layout()
ax.grid(true)
display(fig)

















######################################################

            #Replica symmetry analysis#

######################################################



# time-average per-site spin vector for one trajectory (final tavg_frac window)
# comp=:z or :x
function s_timeavg(sol; comp::Symbol=:z, tavg_frac::Float64=0.2)
    N = sol.p.N
    Nt = size(sol.u, 2)
    tstart = max(1, Int(floor((1.0 - tavg_frac) * Nt)) + 1)
    idx = comp == :x ? (2*N+1 : 3*N) : (4*N+1 : 5*N)
    return vec(mean(sol.u[idx, tstart:Nt]; dims=2))
end

# run one replica: set RNG seed for independent noise, then call your solver
# - p: your System_p (parameters)
# - seed: integer or nothing (if nothing, RNG not reseeded)
# - returns sim (array of Sol or wrapper)
function run_replica(p; seed::Union{Int,Nothing}=nothing, saveat=0.05)
    if seed !== nothing
        Random.seed!(seed)   # ensures independent noise if solver uses global RNG
    end
    # If your many_trajectory_solver takes the param struct and uses p.N_MC to decide # trajectories:
    sim = many_trajectory_solver(p, saveat=saveat)
    return sim
end

# compute overlap between two trajectory objects (choose which trajectory index to use)
# - simA, simB: outputs from run_replica (or arrays of Sol)
# - traj_idxA, traj_idxB: which trajectory within each sim to pick (default 1)
# - comp: :z or :x
# - tavg_frac: final-window averaging fraction
function replica_overlap(simA, simB; traj_idxA::Int=1, traj_idxB::Int=1, comp::Symbol=:z, tavg_frac::Float64=0.2)
    solsA = sols_from_sim(simA)
    solsB = sols_from_sim(simB)
    solA = solsA[traj_idxA]
    solB = solsB[traj_idxB]
    sA = s_timeavg(solA; comp=comp, tavg_frac=tavg_frac)
    sB = s_timeavg(solB; comp=comp, tavg_frac=tavg_frac)
    N = length(sA)
    q = dot(sA, sB) / N
    return q, sA, sB
end

# === Example usage ===
# assume you already have 'p' (System_p) configured exactly as you want
# run two replicas with different seeds so noise is independent
replica1 = run_replica(p; seed=12345, saveat=0.05)
replica2 = run_replica(p; seed=67890, saveat=0.05)

function s_list_from_sim(sim; comp=:z, tavg_frac=0.2)
    sols = sols_from_sim(sim)
    return [ s_timeavg(sol; comp=comp, tavg_frac=tavg_frac) for sol in sols ]
end

# example ensemble pairing (mean overlap across all pairs between the two replicas)
slistA = s_list_from_sim(replica1; comp=:z, tavg_frac=0.2)
slistB = s_list_from_sim(replica2; comp=:z, tavg_frac=0.2)
Overlaps = Float64[]
for a in 1:length(slistA), b in 1:length(slistB)
    push!(Overlaps, dot(slistA[a], slistB[b]) / length(slistA[a]))
end
println("Mean inter-replica overlap = ", mean(Overlaps))




