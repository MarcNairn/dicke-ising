"""THIS SCRIPT SOLVES THE SDE OF THE MAIN TEXT THROUGH A DTWA MONTE CARLO SAMPLING TECHNIQUE
for more info look up documentation on  https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble
"""

# module Main



include("custom_functions.jl")
include("observables.jl")


using Random
using Statistics
using Printf
using DataFrames
using DifferentialEquations 





# -------------------------
# Add minimal Rydberg-dressed functionality
# - add fR (fraction of dressed atoms) to params
# - add V (Ising strength) and rthresh (interaction cutoff)
# - choose dressed spins once per System_p (deterministic given seed)
#   field terms in spin EOM (adds -h * sy to dsx/dt and +h * sx to dsy/dt)
# -------------------------

# helpers for periodic distance
wrap2pi(x) = mod(x, 2pi)

function circ_dist(x1, x2)
    dx = abs(x1 - x2)
    dx = mod(dx, 2pi)
    return min(dx, 2pi - dx)
end


@inline function U_softcore(r, V, Rc)
    x = r / Rc
    return V / (1 + x^6)
end

@inline function dU_softcore_dr(r, V, Rc)
    x = r / Rc
    return -6V * x^5 / (Rc * (1 + x^6)^2)
end

    # Mutable parameter struct
mutable struct System_p
    g::Float64
    ω₀::Float64
    Δc::Float64
    κ::Float64
    temp::Float64
    N::Int
    tspan::Tuple{Float64,Float64}
    N_MC::Int
    # Rydberg/dressed params
    V::Float64
    rthresh::Float64
    dressed_idx::Vector{Int}
    # quench controls (minimal)
    t_quench::Float64      # time at which V is changed (use Inf for "no quench")
    V_after::Float64       # value of V after t_quench
    # preallocated temporaries
    _xwrap::Vector{Float64}
    _ising_field::Vector{Float64}
    _rydberg_force::Vector{Float64}
end

function System_p(g, ω₀, Δc, κ, temp, N, tspan, N_MC, V, rthresh, dressed_idx;
                  t_quench=Inf, V_after=V)
    System_p(g, ω₀, Δc, κ, temp, N, tspan, N_MC, V, rthresh, dressed_idx,
             t_quench, V_after,
             zeros(Float64, N), zeros(Float64, N), zeros(Float64, N))
end


    # # utility to pick dressed spins (stored in p.dressed_idx). deterministic given rng seed.
    # function choose_dressed!(p::System_p; seed::Int = abs(rand(Int)))
    #     rng = MersenneTwister(seed)
    #     N_R = Int(round(p.fR * p.N))
    #     if N_R > 0
    #         # pick a random subset of spins to be dressed
    #         p.dressed_idx = sort(randperm(rng, p.N)[1:N_R])
    #     else
    #         p.dressed_idx = Int[]
    #     end
    #     return p.dressed_idx
    # end

# ----------------------------
w_func(r) = 1.0


function compute_ising_field!(
    ising_field::AbstractVector{Float64},
    xwrap::AbstractVector{<:Real},
    s_z::AbstractVector{<:Real},
    dressed_idx::AbstractVector{<:Integer},
    V::Float64,
    Rc::Float64
)
    N_r = length(dressed_idx)

    @inbounds ising_field .= 0.0
    if N_r <= 1
        return
    end

    @inbounds for a in 1:(N_r-1)
        ia = dressed_idx[a]
        xa = xwrap[ia]
        Pa = 0.5 * (1 + s_z[ia])
        # Pa = 0.5 * s_z[ia]
        for b in (a+1):N_r
            ib = dressed_idx[b]
            xb = xwrap[ib]
            Pb = 0.5 * (1 + s_z[ib])
            # Pb = 0.5 * s_z[ib]
            dx = xa - xb
            dx = mod(dx + π, 2π) - π
            r  = abs(dx)

            # small cutoff
            if r > 1e-12
                U = U_softcore(r, V, Rc)
                ising_field[ia] += U * Pb
                ising_field[ib] += U * Pa
            end
        end
    end
end

function compute_rydberg_force!(
    rydberg_force::AbstractVector{Float64},
    xwrap::AbstractVector{<:Real},
    s_z::AbstractVector{<:Real},
    dressed_idx::AbstractVector{<:Integer},
    V::Float64,
    Rc::Float64;
    r_eps::Float64 = 1e-12
)
    N_r = length(dressed_idx)

    @inbounds rydberg_force .= 0.0
    if N_r <= 1
        return
    end

    @inbounds for a in 1:(N_r-1)
        ia = dressed_idx[a]
        xa = xwrap[ia]
        Pa = 0.5 * (1 + s_z[ia])

        for b in (a+1):N_r
            ib = dressed_idx[b]
            xb = xwrap[ib]
            Pb = 0.5 * (1 + s_z[ib])
            #Pb = 0.5 * s_z[ib]
            dx = xa - xb
            dx = mod(dx + π, 2π) - π
            r  = abs(dx)

            if r > r_eps
                dU = dU_softcore_dr(r, V, Rc)
                f  = -dU * Pa * Pb * sign(dx)

                rydberg_force[ia] += f
                rydberg_force[ib] -= f
            end
        end
    end
end


    ##############################################################################
    ################################SYSTEM FUNCTIONS##############################
    ##############################################################################
    function f_det(du,u,p,t)
    # t is in unit of w_R^{-1}

    # u = [ xⱼ,     pⱼ,      σˣⱼ,       σʸⱼ,      σᶻⱼ,    aᵣ,    aᵢ]
    #     [1:N, N+1:2N, 2N+1..3N, 3N+1..4N, 4N+1..5N, 5N+1, 5N+2]
    # x_j in units of 1/kc (x'_j = kc x_j)
    # p_j in units of hbar*kc (p'_j = p_j/(hbar*kc))

    # ancilla variables

    N::Int = p.N
    g::Float64 = p.g
    aa::Float64 = (u[5N+1]^2 + u[5N+2]^2 - 0.5) # aᵣ² + aᵢ² - 1/2
    Δc::Float64 = p.Δc
    dd::Float64 = 0.0



    # position wrapping
    @inbounds for j in 1:N
        p._xwrap[j] = mod(u[j], 2π)
    end

    s_z_view = @view u[(4N+1):(5N)]

    # Quench time and field 
    Vnow = t < p.t_quench ? p.V : p.V_after

    #gnow = t < p.t_quench ? p.g : p.g_after

    if Vnow == 0.0 || p.rthresh==0
        p._ising_field .= 0.0
        p._rydberg_force .= 0.0
    else
        compute_ising_field!(p._ising_field, p._xwrap, s_z_view, p.dressed_idx, Vnow, p.rthresh)
        compute_rydberg_force!(p._rydberg_force, p._xwrap, s_z_view, p.dressed_idx, Vnow, p.rthresh)
    end


    @inbounds for j in 1:N
        sinuj, cosuj = sincos(u[j])

        dd += (g * u[2N+j]) * cosuj
        
        h_ising = p._ising_field[j]
        f_ryd = p._rydberg_force[j]

        du[j] = 2u[N+j]
        du[N+j] = sinuj * 2*g * u[2N+j] * u[5N+1] - 1/2 * f_ryd 

        du[2N+j] = -u[3N+j] * p.ω₀ - h_ising * u[3N+j]
        du[3N+j] = u[2N+j] * p.ω₀ - 4*cosuj * u[4N+j] * (g * u[5N+1]) + h_ising * u[2N+j]
        du[4N+j] = 4cosuj * (g * u[3N+j] * u[5N+1])
    end

    du[5N+1] = Δc * u[5N+2] - p.κ * u[5N+1]
    du[5N+2] = -Δc * u[5N+1] + dd/2 - p.κ * u[5N+2]
end

    ###Stochastic part of the SDE (cavity noise alone)####

function f_noise(du,u,p,t)
    N::Int = p.N
    du[1:5N] .= 0.0
    du[5N+1] = sqrt((1/2)*p.κ)
    du[5N+2] = sqrt((1/2)*p.κ)
end

###INITIAL CONDITIONS###

    
    function initial_conditions(p::System_p, seed=abs(rand(Int)))
        N::Int = p.N
        Random.seed!(seed) # random number generator
        u0 = zeros(5N + 2)

        u0[1:N] = 2pi.*rand(N) # generate random positions
        # Place atoms equidistantly
        # for j in 1:N
        #     u0[j] = 2pi*(j-1)/N
        # end
        u0[N+1:2N] = p.temp .* randn(N) # generate random momenta, temp refers to the square root of a temperature unit

        u0[2N+1:4N] = 2bitrand(2N) .- 1  # σˣⱼ and σʸⱼ are 1 or -1  
        # for j in 4N+1:5N
        #     u0[j] = (-1)^(j-1) # Neel order
        # end
        u0[4N+1:5N] .=  -1 # σᶻⱼ = -1, atoms in the ground state

        u0[5N+1:end] .= 0 # cavity empty
        return u0
    end

    #@everywhere 
    function define_prob_from_parameters(p::System_p,seed=abs(rand(Int)))
        # initial conditions
        Random.seed!(seed) # random number generator
        u0 = initial_conditions(p,abs(rand(Int)))
        u0_arr = [initial_conditions(p,abs(rand(Int))) for j=1:p.N_MC]

    # NEED TO DEFINE THE FUNCTION TO REDRAW INITIAL CONDITIONS AT EVERY TRAJECTORY
        function prob_func(prob,i,repeat)
            # @. prob.u0 = initial_conditions(N,κ,rng)
            @. prob.u0 = u0_arr[i]
        prob
        end

        prob = SDEProblem(f_det,f_noise,u0,p.tspan,p)
        monte_prob = EnsembleProblem(prob, prob_func = prob_func)

        return prob, monte_prob
    end




    function many_trajectory_solver(p::System_p;saveat::Float64=10.0,seed::Int=abs(rand(Int)),maxiters::Int=Int(1e9))#, reltol::Float64=1e-4)
        prob, monte_prob = define_prob_from_parameters(p,seed)
        #print("calculating $(p.N_MC) trajectories on $(gethostname()) with $(nworkers()) workers..")
        elt = @elapsed sim = solve(monte_prob::EnsembleProblem, SOSRA2(), EnsembleThreads(), trajectories=p.N_MC, saveat=saveat, maxiters=maxiters, progress=true)#; dt = 1e-5)

        # EnsembleDistributed() recommended here when each trajectory is not very quick (like here)
        println("done in $elt seconds.")
        return sim
    end
    


#######################################
#        POST PROCESSING              #   
#######################################

function get_last_steps(sol::Sol)
    Sol(sol.u[:,end-1:end],sol.p,sol.t[end-1:end],sol.alg)
end

function get_last_steps(sim::Array{Sol,1})
    [get_last_steps(sol) for sol in sim]
end

function regenerate_prob(sol::Sol,deltat::Real)
    u0 = sol.u[:,end]
    p = sol.p
    tspan = (sol.t[end], sol.t[end]+deltat)
    prob = SDEProblem(f_det,f_noise,u0,tspan,p)
end

function regenerate_prob(sim::Array{Sol,1},deltat::Real)
    prob =  regenerate_prob(sim[1],deltat)
    function prob_func(prob,i,repeat)
        regenerate_prob(sim[i],deltat)
    end

    monte_prob = EnsembleProblem(prob,prob_func=prob_func)
end


function stringfromp(p)
    U₁,U₂,S₁,S₂,ω₀,Δc,κ,N = parsfromp(p)
    str1 = @sprintf("U1%.4f_U2%.4f_S1%.4f%+.4fim_S2%.4f%+.4fim_De%.4f_Dc%.4f_k%.4f_N%d", U₁,U₂,real(S₁),imag(S₁),real(S₂),imag(S₂),ω₀,Δc,κ,N)
end

function pfromstring(str::String)
    x = split(str,'_')
    U₁ = parse(Float64,x[1][3:end])
    U₂ = parse(Float64,x[2][3:end])
    S₁ = parse(Complex{Float64},x[3][3:end])
    S₂ = parse(Complex{Float64},x[4][3:end])
    ω₀ = parse(Float64,x[5][3:end])
    Δc = parse(Float64,x[6][3:end])
    κ = parse(Float64,x[7][2:end])
    N = parse(Int,x[8][2:end])
    return U₁,U₂,S₁,S₂,ω₀,Δc,κ,N
end

function join_trajectories(sim::Array{Sol,1})
    idx = size(sim[1].t)[1]
    join_trajectories(sim,idx)
end

function join_trajectories(sim::Array{Sol,1},idx)
    N::Int = try sim[1].p.N catch; sim[1].p[10] end # for backward compatibility

    ntraj = size(sim)[1]

    u0 = zeros(ntraj*size(sim[1].u)[1])

    for i in 1:ntraj
        u0[range((i-1)*N+1,length=N)] = sim[i].u[1:N,idx]
        u0[range((ntraj+i-1)*N+1,length=N)] = sim[i].u[N+1:2N,idx]
        u0[range((2ntraj+i-1)*N+1,length=N)] = sim[i].u[2N+1:3N,idx]
        u0[range((3ntraj+i-1)*N+1,length=N)] = sim[i].u[3N+1:4N,idx]
        u0[range((4ntraj+i-1)*N+1,length=N)] = sim[i].u[4N+1:5N,idx]
        u0[range(5ntraj*N+(i-1)*2+1,length=2)] = sim[i].u[5N+1:5N+2,idx]
    end
    u0
end

function sim2df(sim::Array{Sol,1})
    nvars = size(sim[1].u)[1]
    N::Int = (nvars-2)/5

    value_names = Symbol[]
    for var in ["x_","p_","sx_","sy_","sz_"]
        for i = 1:N
            push!(value_names,Symbol(var*"$i"))
        end
    end
    push!(value_names, :a_r)
    push!(value_names, :a_i)
    # push!(value_names, :timestamp)
    # push!(value_names, :traj)

    dfs = DataFrame[]
    for (k,sol) in enumerate(sim)
        dict = Dict{Symbol,Any}(value_names[i] => sol.u[i,:] for i = 1:nvars)
        dict[:a_r] = sol.u[5N+1,:]
        dict[:a_i] = sol.u[5N+2,:]
        for (o_, o) in observable_dict
            dict[o_] = expect(o,sol)
        end
        dict[:timestamp] = sol.t
        dict[:traj] = k
        push!(dfs,DataFrame(dict))
    end
    vcat(dfs...)
end


############ HELPERS ###########

function categorize_traj(sol::Sol) #categorise the broken Z₂ symmetry X (Y) states
    expect(X,sol)[end] >=0 ? x=1 : x=-1
    return x
end

function categorize_traj(sol::Sol, obs::Observable) # Observable-dependent choice
    expect(obs,sol)[end] >=0 ? x=1 : x=-1
    return x
end



function categorize_traj(sim::Array{Sol,1})

    plus_sols = Sol[]
    minus_sols = Sol[]
    other_sols = Sol[]

    for sol in sim
        cat = categorize_traj(sol)
        if cat == 1
            push!(plus_sols, sol)
        elseif cat == -1
            push!(minus_sols, sol)
        else
            push!(other_sols, sol)
        end
    end

    return plus_sols, minus_sols, other_sols
end


function categorize_traj(sim::Array{Sol,1}, obs::Observable)

    plus_sols = Sol[]
    minus_sols = Sol[]
    other_sols = Sol[]

    for sol in sim
        cat = categorize_traj(sol, obs)
        if cat == 1
            push!(plus_sols, sol)
        elseif cat == -1
            push!(minus_sols, sol)
        else
            push!(other_sols, sol)
        end
    end

    return plus_sols, minus_sols, other_sols
end
# end # module