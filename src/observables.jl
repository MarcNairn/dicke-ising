""" 

Defininition of all observables of parent module via expectation values over stochastic trajectories


"""

using LaTeXStrings

##################################################################################

############################## OBSERVABLES and HELPERS ###########################

##################################################################################


### Observable structs

Base.@kwdef struct Observable
    s_traj
    formula::String
    short_name::String = formula
    name::String = short_name
    # params::Dict = Dict()  
end


Observable(s_traj,formula,short_name) = Observable(s_traj=s_traj,formula=formula,short_name=short_name)
Observable(s_traj,formula) = Observable(s_traj=s_traj,formula=formula)

function expect(o::Observable,sol::RODESolution)
    sol_ = extract_solution(sol)[1]
    expect(o,sol_)
end

function expect(o::Observable,sim::EnsembleSolution)
    sim_ = extract_solution(sim)
    expect(o,sim_)
end

function expect(o::Observable,sol::Sol)
    # Merge any params passed to expect with params in the Observable
    # all_params = merge(o.params, params)
    o.s_traj(sol)#, all_params)
end

function expect_full(o::Observable, sim::Array{Sol,1})#; params=Dict())
    [expect(o, sim[j]) for j=1:length(sim)]
end

function expect(o::Observable, sim::Array{Sol,1})#; params=Dict())
    Os = expect_full(o, sim)#; params=params)
    Omean = mean(Os)
    Ostd = stdm(Os, Omean)
    bb = hcat(Os...)
    Oq90 = [quantile(bb[i,:], [0.05, 0.95]) for i in 1:size(bb)[1]]

    return Omean, Ostd, Oq90
end


### Define observables of interest below

# kinetic energy/N
function traj_kinetic_energy(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end # for backward compatibility
    [mean(sol.u[N+1:2N,j].^2) for j=1:length(sol.t)]
end
Ekin = Observable(traj_kinetic_energy,L"\sum_j p_j^2",L"E_\mathrm{kin}","kinetic energy")

# kurtosis of momentum distribution
function traj_kurt(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end # for backward compatibility
    [mean(sol.u[N+1:2N,j].^4)./mean(sol.u[N+1:2N,j].^2).^2 for j=1:length(sol.t)]
end
kurt = Observable(traj_kurt,L"\sum_j p_j^4 / (p^2)^2",L"\mathrm{kurtosis}","kurtosis")

# X = ∑ⱼσₓʲ cos(xⱼ)/N
function traj_X(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [mean(sol.u[2N+1:3N,j].*cos.(sol.u[1:N,j])) for j=1:length(sol.t)]
end
X = Observable(traj_X,L"N^{-1}\sum_j \sigma_x^j \cos(x_j)",L"X",L"order parameter $X$")

# absX = |∑ⱼσₓʲ cos(xⱼ)|/N
function traj_absX(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [abs(mean(sol.u[2N+1:3N,j].*cos.(sol.u[1:N,j]))) for j=1:length(sol.t)]
end
absX = Observable(traj_absX,L"N^{-1}\abs{\sum_j \sigma_x^j \cos(x_j)}",
                  L"\abs{X}",L"order parameter $\abs{X}$")

function traj_X2(sol::Sol)
N::Int = try sol.p.N catch; sol.p[10] end
[mean((sol.u[2N+1:3N,j].*cos.(sol.u[1:N,j]))).^2 for j=1:length(sol.t)]
end
X2 = Observable(traj_X2,L"N^{-1}\sum_j (\sigma_x^j \cos(x_j))^2",L"X",L"order parameter $X^2$")

# Y = ∑ⱼσ\_yʲ cos(xⱼ)/N
function traj_Y(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [mean(sol.u[3N+1:4N,j].*cos.(sol.u[1:N,j])) for j=1:length(sol.t)]
end
Y = Observable(traj_Y,L"N^{-1}\sum_j \sigma_y^j \cos(x_j)",L"Y",L"order parameter $Y$")

# absY = |∑ⱼσ\_yʲ cos(xⱼ)|/N
function traj_absY(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [abs(mean(sol.u[3N+1:4N,j].*cos.(sol.u[1:N,j]))) for j=1:length(sol.t)]
end
absY = Observable(traj_absY,L"N^{-1}\abs{\sum_j \sigma_y^j \cos(x_j)}",
                  L"\abs{Y}",L"order parameter $\abs{Y}$")

# Z =  ∑ⱼσ\_zʲ⋅cos(xⱼ)/N
function traj_Z(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [mean(sol.u[4N+1:5N,j].*cos.( sol.u[1:N,j])) for j=1:length(sol.t)]
end
Z = Observable(traj_Z,L"N^{-1}\sum_j\sigma_z\cos(x_j)",L"Z")

# |Z| =  |∑ⱼσ\_zʲ⋅cos(xⱼ)|/N
function traj_absZ(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [abs(mean(sol.u[4N+1:5N,j].*cos.(sol.u[1:N,j]))) for j=1:length(sol.t)]
end
absZ = Observable(traj_absZ,L"N^{-1}\abs{\sum_j\sigma_z\cos(x_j)}",
                       L"\abs{Z}")

# sigmax = ∑ⱼσₓʲ/N
function traj_sigmax(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [mean(sol.u[2N+1:3N,j]) for j=1:length(sol.t)]
end
Sx = Observable(traj_sigmax,L"N^{-1}\sum_j \sigma_x^j",L"$\langle \hat J_x \rangle$",L"\sigma_x")

# absSx = |∑ⱼσₓʲ|/N
function traj_absSx(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [abs(mean(sol.u[2N+1:3N,j])) for j=1:length(sol.t)]
end
absSx = Observable(traj_absSx,L"N^{-1}\abs{\sum_j \sigma_x^j}",L"\abs{\sigma_x}")

# sigmay = ∑ⱼσ\_yʲ/N
function traj_sigmay(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [mean(sol.u[3N+1:4N,j]) for j=1:length(sol.t)]
end
Sy = Observable(traj_sigmay,L"N^{-1}\sum_j \sigma_y^j",L"$\langle \hat J_y \rangle$",L"\sigma_y")

# absSy = |∑ⱼσ\_yʲ|/N
function traj_absSy(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [abs(mean(sol.u[3N+1:4N,j])) for j=1:length(sol.t)]
end
absSy = Observable(traj_absSy,L"N^{-1}\abs{\sum_j \sigma_y^j}",L"\abs{\sigma_y}")


# sigmaz = ∑ⱼσ\_zʲ/N
function traj_sigmaz(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [mean(sol.u[4N+1:5N,j]) for j=1:length(sol.t)]
end
Sz = Observable(traj_sigmaz,L"N^{-1}\sum_j \sigma_z^j",L"$\langle \hat J_z \rangle$",L"\sigma_z")

# absSz = |∑ⱼσ\_zʲ|/N
function traj_absSz(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [abs(mean(sol.u[4N+1:5N,j])) for j=1:length(sol.t)]
end
absSz = Observable(traj_absSz,L"N^{-1}\abs{\sum_j \sigma_z^j}",L"\abs{\sigma_z}")

# sigmax² = ∑ⱼ(σₓʲ)²/N
function traj_sigmax2(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [mean(sol.u[2N+1:3N,j].^2) for j=1:length(sol.t)]
end
Sx2 = Observable(traj_sigmax2,L"N^{-1}\sum_j (\sigma_x^j)^2",L"\sigma_x^2")

# sigmay² = ∑ⱼ(σ\_yʲ)²/N
function traj_sigmay2(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [mean(sol.u[3N+1:4N,j].^2) for j=1:length(sol.t)]
end
Sy2 = Observable(traj_sigmay2,L"N^{-1}\sum_j (\sigma_y^j)^2",L"\sigma_y^2")

# sigmaz² = ∑ⱼ(σ\_zʲ)²/N
function traj_sigmaz2(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [mean(sol.u[4N+1:5N,j].^2) for j=1:length(sol.t)]
end
Sz2 = Observable(traj_sigmaz2,L"N^{-1}\sum_j (\sigma_z^j)^2",L"\sigma_z^2")

# cos(x)^2 = ∑ⱼcos(xⱼ)²/N
function traj_cos2(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [mean(cos.(sol.u[1:N,j]).^2) for j=1:length(sol.t)]
end
Cos2 = Observable(traj_cos2,L"N^{-1}\sum_j\cos(x_j)^2",L"\mathcal{B}",L"\cos^2")

# cos(x) = ∑ⱼcos(xⱼ)/N
function traj_cos(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [mean(cos.(sol.u[1:N,j])) for j=1:length(sol.t)]
end
Cos = Observable(traj_cos,L"N^{-1}\sum_j\cos(x_j)",L"\cos")

# cavity population = aᵣ² + aᵢ² - 0.5
function traj_adaga(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [sum(sol.u[5N+1:5N+2,j].^2) for j=1:length(sol.t)]
end
adaga = Observable(traj_adaga,L"a_r^2+a_i^2-0.5",L"\langle a^\dag a\rangle","cavity population")

# real part cavity field aᵣ
function traj_ar(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [sol.u[5N+1,j] for j=1:length(sol.t)]
end
ar = Observable(traj_ar, L"\langle \frac{a+a^\dag}{2}\rangle", L"$\langle \hat{a}_{\mathrm{r}} \rangle$", "real part of cavity field")

# absolute value of real part cavity field aᵣ
function traj_absar(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [abs(sol.u[5N+1,j]) for j=1:length(sol.t)]
end
absar = Observable(traj_absar, L"\abs{a_r}", L"\langle \frac{\abs{a+a^\dag}}{2}\rangle", "absolute value of real part of cavity field")

# imaginary part cavity field aᵢ
function traj_ai(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [sol.u[5N+2,j] for j=1:length(sol.t)]
end
ai = Observable(traj_ai, L"\langle i\frac{a-a^\dag}{2}\rangle", L"$\langle \hat{a}_{\mathrm{r}} \rangle$", "imaginary part of cavity field")

# absolute value of imaginary part cavity field aᵢ
function traj_absai(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [abs(sol.u[5N+2,j]) for j=1:length(sol.t)]
end
absai = Observable(traj_absai, L"\abs{a_i}", L"\langle \frac{\abs{a-a^\dag}}{2}\rangle", "absolute value of imaginary part of cavity field")

# adiabatic cavity field aᵣ
function traj_adiabaticar(sol::Sol)
    N::Int = sol.p.N
    S₁::Complex{Float64} = sol.p.S₁
    S₂::Complex{Float64} = sol.p.S₂
    U₁::Float64 = sol.p.U₁
    U₂::Float64 = sol.p.U₂
    Δc::Float64 = sol.p.Δc
    κ::Float64 = sol.p.κ
    [real(sum((S₁.*(sol.u[2N+1:3N,j]-sol.u[3N+1:4N,j]*im)./2 + S₂.*(sol.u[2N+1:3N,j]+sol.u[3N+1:4N,j]*im)./2).*cos.(sol.u[1:N,j]))/(Δc*im-κ)) for j=1:length(sol.t)]
end
adiabaticar = Observable(traj_adiabaticar, L"\tilde{a}", L"\langle \tilde{a_r} \rangle", "adiabatic value of the real part of the cavity field")

# adiabatic cavity population
function traj_adiabaticadaga(sol::Sol)
    N::Int = sol.p.N
    S₁::Complex{Float64} = sol.p.S₁
    S₂::Complex{Float64} = sol.p.S₂
    U₁::Float64 = sol.p.U₁
    U₂::Float64 = sol.p.U₂
    Δc::Float64 = sol.p.Δc
    κ::Float64 = sol.p.κ
    [abs2(sum((S₁.*(sol.u[2N+1:3N,j]-sol.u[3N+1:4N,j]*im)./2 + S₂.*(sol.u[2N+1:3N,j]+sol.u[3N+1:4N,j]*im)./2).*cos.(sol.u[1:N,j]))/(Δc*im-κ)) for j=1:length(sol.t)]
end
adiabaticadaga = Observable(traj_adiabaticadaga, L"\tilde{a}", L"\langle \tilde{a^\dag a} \rangle", "adiabatic value of the cavity population")

# single_sigmax² = (σₓʲ)²
function traj_single_sx2(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [(sol.u[2N+Int(round((N/2))),j].^2) for j=1:length(sol.t)]
end
single_sx2 = Observable(traj_single_sx2,L"(\sigma_x)^2",L"\sigma_x^2")

function traj_single_sy2(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [(sol.u[3N+Int(round((N/2))),j].^2) for j=1:length(sol.t)]
end
single_sy2 = Observable(traj_single_sy2,L"(\sigma_y)^2",L"\sigma_y^2")

function traj_single_sz2(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [(sol.u[4N+Int(round((N/2))),j].^2) for j=1:length(sol.t)]
end
single_sz2 = Observable(traj_single_sz2,L"(\sigma_z)^2",L"\sigma_z^2")


function traj_phase(sol::Sol)
    N::Int = try sol.p.N catch; sol.p[10] end
    [angle(sol.u[5N+1,j] + im*sol.u[5N+2,j]) for j=1:length(sol.t)]
end

a_phase = Observable(traj_phase,L"\arg(a)",L"\arg(a)","cavity phase"
)

#####################################################################
####################### Structure factors ###########################
#####################################################################

using FFTW

# compute S_z(q,ω) spectrum along a single trajectory
function traj_DSSF_Z_fft(sol::Sol, q::Float64)
    N::Int = sol.p.N

    # time series of S_z(q,t)
    Sq_t = [mean(sol.u[4N+1:5N,j] .* exp.(im*q .* sol.u[1:N,j])) for j=1:length(sol.t)]
    Sq_w = fft(Sq_t)
    # spectral weight 
    abs2.(Sq_w)
end

function DSSF_Z_fft(q::Float64)
    Observable(
        sol -> traj_DSSF_Z_fft(sol,q),
        L"|S_z(q,\omega)|^2",
        L"S(q,\omega)",
        "dynamical spin structure factor z"
    )
end

# Equivalent for Sx

# compute S_x(q,ω)
function traj_DSSF_X_fft(sol::Sol, q::Float64)
    N::Int = sol.p.N

    # time series of S_x(q,t)
    Sq_t = [mean(sol.u[2N+1:3N,j] .* exp.(im*q .* sol.u[1:N,j])) for j=1:length(sol.t)]
    Sq_w = fft(Sq_t)
    # spectral weight 
    abs2.(Sq_w)
end

function DSSF_X_fft(q::Float64)
    Observable(
        sol -> traj_DSSF_X_fft(sol,q),
        L"|S_x(q,\omega)|^2",
        L"S(q,\omega)",
        "dynamical spin structure factor x"
    )
end


function traj_DSSF_Y_fft(sol::Sol, q::Float64)
    N::Int = sol.p.N

    # time series of S_y(q,t)
    Sq_t = [mean(sol.u[3N+1:4N,j] .* exp.(im*q .* sol.u[1:N,j])) for j=1:length(sol.t)]
    Sq_w = fft(Sq_t)
    # spectral weight 
    abs2.(Sq_w)
end

function DSSF_Y_fft(q::Float64)
    Observable(
        sol -> traj_DSSF_Y_fft(sol,q),
        L"|S_y(q,\omega)|^2",
        L"S(q,\omega)",
        "dynamical spin structure factor y"
    )
end
