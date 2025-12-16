""" 

Main program to sample initial conditions and simulate single spin and cavity dynamics for a system of N spins coupled to a single mode cavity via Dicke interaction. 
Of these nt = fN (0 ≤ f ≤ 1 ) also experience a nearest neighbour Ising like interaction when their position comes within the potential V(xⱼ) created by the nearest dressed atom 

"""

using PyPlot

######################
""" Call main source code functions from helper """

include("src.jl");
######################

N = 8;
n_t = 3;
g0 = 15.0;
Vx = 1;
J=0;
κ = 10;
ω0 = 2;
Δc = -10;
pj = 1;
p = System_p(
    0.0, 0.0,                 # U1, U2
    Complex(g0,0.0),         # S1 = g0
    Complex(g0,0.0),         # S2 = g0
    ω0,                     # ω₀
    Δc,                     # Δc
    κ,                     # κ
    pj,                     # pj
    N,
    (0.0, 10.0),            # tspan
    10,                       # N_MC
    J,                     # J
    Vx,                     # Vx
    0.25 * Vx,              # sigma_chi
    n_t                     # n_t
)


seed = 123456789;
u0, dressed_mask = initial_conditions(p, seed);
global CURRENT_DRESSED_MASK = dressed_mask;
println("Dressed indices: ", findall(dressed_mask))


prob = SDEProblem(f_det, f_noise, u0, p.tspan, p);

sol = solve(prob, SOSRA2(), reltol=1e-3, abstol=1e-4, dt=1e-2, saveat=0.5, maxiters=1e8)


    """ 
Data analysis
"""


function unpack_state_column(y, p::System_p)
    N = p.N
    x = y[1:N]
    pvec = y[N+1:2N]
    mx = y[2N+1:3N]
    my = y[3N+1:4N]
    mz = y[4N+1:5N]
    ar = y[5N+1]; ai = y[5N+2]
    return x, pvec, mx, my, mz, ar, ai
end

times = sol.t
T = length(times)
alpha_abs = zeros(T)
mz_dressed = zeros(T)
mz_undressed = zeros(T)

for ti in 1:T
    y = sol.u[ti]
    x, pvec, mx, my, mz, ar, ai = unpack_state_column(y, p)
    alpha_abs[ti] = sqrt(ar^2 + ai^2)
    mz_dressed[ti] = mean(mz[dressed_mask])
    mz_undressed[ti] = mean(mz[.!dressed_mask])
end

figure()
plot(times, alpha_abs, "k-")
xlabel(L"t")
ylabel(L"|\alpha|")
tight_layout()
display(gcf())


figure()
plot(times, mz_dressed, "b-", label=L"\langle\sigma_z\rangle_{\text{D}}")
plot(times, mz_undressed, "r--", label=L"\langle\sigma_z\rangle_{\text{U}}")
xlabel(L"t", fontsize=14)
ylabel(L"\langle\sigma_z\rangle", fontsize=14)
legend(fontsize=12)
tight_layout()
display(gcf())


# compute C_Ising(t)
function compute_chi_matrix(x, mask, Vx, sigma)
    N = length(x)
    dx = similar(x, N, N)
    # Compute periodic distances
    for i in 1:N, j in 1:N
        dx_ij = x[i] - x[j]
        # Apply periodic boundary: mod into [-π, π]
        dx_ij = mod(dx_ij + π, 2π) - π
        dx[i,j] = dx_ij
    end
    r = abs.(dx)
    W = W_of_r(r, Vx, sigma)
    for i in 1:N
        W[i,i] = 0.0
    end
    svec = Float64.(mask)
    return (svec .* svec') .* W
end



times = sol.t
T = length(times)
C_Ising = zeros(T)
idx = findall(dressed_mask)
n_t = length(idx)

if n_t == 0
    @warn "No dressed atoms (n_t == 0). C_Ising will be zero."
else
    for ti in 1:T
        y = sol.u[ti]
        x, pvec, mx, my, mz, ar, ai = unpack_state_column(y, p)
        chi = compute_chi_matrix(x, dressed_mask, p.Vx, p.sigma_chi)
        chi_sub = chi[idx, idx]
        mz_sub = mz[idx]
        Cval = sum(chi_sub .* (mz_sub * transpose(mz_sub)))   # sum_{i,j in D} chi_ij mz_i mz_j
        C_Ising[ti] = Cval / (n_t^2)                          # normalized by n_t^2
    end
end



figure(figsize=(6,3))
plot(times, C_Ising, "m-")
xlabel(L"t")
ylabel(L"C_{\mathrm{Ising}}(t)")
tight_layout()
display(gcf())