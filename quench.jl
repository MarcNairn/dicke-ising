include("src/main.jl")
include("src/plotting.jl")


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


g = 1
ω₀ = 20.0
Δc = -100.0
κ = 100.0
temp = 20.0
N = 100
tspan = (0,1)
N_MC = 30

V = 0 # Ising strength between dressed atoms
rthresh = 0.1*π # cutoff distance for chi (in same units as x mod 2π)

sqrt.(-1 * (κ^2 + Δc^2) / (2 * Δc) / N) * sqrt.(g)

dressed_idx = randperm(N)[1:N]; # Does nothing for now

p = System_p(g, ω₀, Δc, κ, temp, N, tspan, N_MC, V, rthresh, dressed_idx)

sim = many_trajectory_solver(p, saveat=0.01)


V_new = 5;
p_new  = System_p(g, ω₀, Δc, κ, temp, N, tspan, N_MC, V_new, rthresh, dressed_idx);
deltat = 1;

quench = quench_prob(extract_solution(sim), p_new, deltat)

sim_q =  solve(quench, SOSRA2(), EnsembleThreads(), trajectories=p.N_MC, saveat=0.01, progress=true)


plus, minus, _ = categorize_traj(extract_solution(sim))
plot_spinspositionhisto(plus)

plot_cavity_wigner(sim_q)

######### COMBINED SIMULATIONS #########


# simT = merge_sim(extract_solution(sim), extract_solution(sim_q))

# plus, minus, _ = categorize_traj(extract_solution(simT))
# plot_spinspositionhisto(extract_solution(plus))


mean_X_pre, std_X_pre, q90_X_pre = expect(obs_X2, sim)
mean_X_post, std_X_post, q90_X_post = expect(obs_X2, sim_q)


q90_X_pre = hcat(q90_X_pre...)
q90_X_post = hcat(q90_X_post...)


t_pre = sim[1].t
t_post = sim_q[1].t  # Shift post-quench time


fig, ax  = subplots(figsize=(5,3))
plot(t_pre, mean_X_pre, lw=3, color="teal")
fill_between(t_pre, q90_X_pre[1,:], q90_X_pre[2,:], alpha=0.25, color="lightblue")
plot(t_post, mean_X_post, lw=3, color="peru")
fill_between(t_post, q90_X_post[1,:], q90_X_post[2,:], alpha=0.25, color="peru")
axvline(x=deltat, color="black", linestyle="--", alpha=0.7, label="Quench")
xlabel(L"t\omega_R"); ylabel(L"\langle X^2\rangle");
legend()
display(fig)


#################################

mean_C_pre, std_C_pre, q90_C_pre = expect(obs_xcorr, sim)
mean_C_post, std_C_post, q90_C_post = expect(obs_xcorr, sim_q)

q90_C_pre = hcat(q90_C_pre...)
q90_C_post = hcat(q90_C_post...)


t_pre = sim[1].t
t_post = sim_q[1].t  # Shift post-quench time


fig, ax  = subplots(figsize=(5,3))
plot(t_pre, mean_C_pre, lw=3, color="teal")
fill_between(t_pre, q90_C_pre[1,:], q90_C_pre[2,:], alpha=0.25, color="lightblue")
plot(t_post, mean_C_post, lw=3, color="peru")
fill_between(t_post, q90_C_post[1,:], q90_C_post[2,:], alpha=0.25, color="peru")
axvline(x=deltat, color="black", linestyle="--", alpha=0.7)
xlabel(L"t\omega_R"); ylabel(L"\langle C_{ij}^x\rangle");
# legend()
display(fig)