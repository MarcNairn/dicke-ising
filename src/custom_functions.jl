
## Adapted code structure from spin self organization with core function calls


using DifferentialEquations: EnsembleSolution, RODESolution, ODESolution

export extract_solution, save_datal, load_datal, Sol,merge_sol, merge_sim
export intervallize_array, load_datall
export yourpart

struct Sol
    u::Array{Float64,2}
    p
    t::Array{Float64,1}
    alg::String
end

Sol(u,p,t) = Sol(u,p,t,"")

function extract_solution(sol::ODESolution)
    [Sol(hcat(sol.u...),sol.prob.p,sol.t,repr(sol.alg))]
end

function extract_solution(sol::RODESolution)
    [Sol(hcat(sol.u...),sol.prob.p,sol.t,repr(sol.alg))]
end

function extract_solution(sim::EnsembleSolution)
    trajectories = Array{Sol,1}()
    for sol in sim
        append!(trajectories,extract_solution(sol))
    end
    trajectories
end

function extract_solution(sim::Array{Sol,1})
    sim
end

function sols_from_sim(sim) ### Backup call with failsafe
    try
        return extract_solution(sim)
    catch
        return sim
    end
end

# use vcat to merge Array{Sol,1}

function merge_sim(sim1::Array{Sol,1}...)
    vcat(sim1...)
end

function merge_sol(sol1::Sol,sol2::Sol)
    # perform checks
    if sol1.p != sol2.p
        error("solutions do not have same parameters")
    end
    if sol1.u[:,end] != sol2.u[:,1]
        error("endpoint in sol1 differ from initial point int sol2")
    end
    t = vcat(sol1.t,sol2.t[2:end])
    Sol(hcat(sol1.u,sol2.u[:,2:end]),sol1.p,t,sol1.alg*sol2.alg)
end

function merge_sol(sol1::Sol,sol2::Sol,sols::Sol...)
    sol = merge_sol(sol1,sol2)
    for x in sols
        sol = merge_sol(sol,x)
    end
    sol
end

function merge_sol(sim1::Array{Sol,1},sim2::Array{Sol,1})
    if size(sim1) != size(sim2)
        error("sim1 and sim2 must have the same size")
    end
    sim = Sol[]
    for i in 1:size(sim1)[1]
        push!(sim, merge_sol(sim1[i],sim2[i]))
    end
    sim
end

function merge_sol(sim1::Array{Sol,1},sim2::Array{Sol,1},sims::Array{Sol,1}...)
    sim = merge_sol(sim1,sim2)
    for x in sims
        sim = merge_sol(sim,x)
    end
    sim
end


function keep_input_time(sim::Array{Sol,1}, required_t::Int)
    sorted_sim = Sol[]
    for i in 1:length(sim)
        if sim[i].t[end]==required_t
            push!(sorted_sim, sim[i])
        end
    end
    return sorted_sim
end


