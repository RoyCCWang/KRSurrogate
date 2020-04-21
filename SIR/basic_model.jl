
#### SIR.

function evalposteriorSIR(θ::Vector{T},
                          const_θ::Vector{T},
                          parserfunc::Function,
                          Σ_y::Matrix{T},
                          y_𝓣::Vector{T},
                          𝓣;
                          t0::T = 0.0,
                          t_end::T = 365.0,
                          β_prior_shape::T = 1.0,
                          β_prior_scale::T = 1.0,
                          γ_prior_shape::T = 1.0,
                          γ_prior_scale::T = 1.0) where T <: Real
    #
    k, I_0, S_0, R_0, β, γ, N_pop = parserfunc(θ, const_θ)

    out = evalSIRlnpdf( Σ_y, k, I_0, S_0, R_0,
                        β, γ, N_pop, y_𝓣, 𝓣;
                          t0 = t0,
                          t_end = t_end,
                          β_prior_shape = β_prior_shape,
                          β_prior_scale = β_prior_scale,
                          γ_prior_shape = γ_prior_shape,
                          γ_prior_scale = γ_prior_scale)

    return out
end

"""
Evaluates the joint probability density of θ and y(𝓣).
y_𝓣 is the observation of y(𝓣).
"""
function evalSIRlnpdf(   Σ_y::Matrix{T},
                        k::T,
                        I_0::T,
                        S_0::T,
                        R_0::T,
                        β::T,
                        γ::T,
                        N::T,
                        y_𝓣::Vector{T},
                        𝓣;
                        t0::T = 0.0,
                        t_end::T = 365.0,
                        β_prior_shape::T = 1.0,
                        β_prior_scale::T = 1.0,
                        γ_prior_shape::T = 1.0,
                        γ_prior_scale::T = 1.0) where T <: Real
    ### set up priors.
    dist_β = Distributions.Gamma(β_prior_shape, β_prior_scale)
    dist_γ = Distributions.Gamma(γ_prior_shape, γ_prior_scale)

    # ## debug.
    # println("β = ", β)
    # println("γ = ", γ)
    # println("S_0 = ", S_0)
    # println("I_0 = ", I_0)
    # println("R_0 = ", R_0)
    # println("k = ", k)
    # println("N = ", N)



    # # draw. This is for MCMC.
    # β = rand(dist_β)
    # γ = rand(dist_γ)

    # evaluate the prior portion of the joint pdf.
    ln_p_β = Distributions.logpdf(dist_β, β)
    ln_p_γ = Distributions.logpdf(dist_γ, γ)

    ### set up likelihood.
    S_t_unused, I_t,
      R_t_unused = solveSIRnumerical(  I_0,
                                        S_0,
                                        R_0,
                                        β,
                                        γ,
                                        N;
                                        t0 = t0,
                                        t_end = t_end)


    # evaluate the likelihood portion.
    μ = k .* I_t.(𝓣)
    dist_y = Distributions.MvNormal(μ, Σ_y)
    #ln_likelihood = Distributions.logpdf(dist_y, y_𝓣)

    res = y_𝓣-μ
    unnormalized_ln_likelihood = -dot(res, Σ_y\res)/2

    n_st = 3
    n_end = 4
    res = res[n_st:n_end]
    unnormalized_ln_likelihood = -dot(res, Σ_y[n_st:n_end,n_st:n_end]\res)/2

    ###

    ## actual probability. too small for density fit.
    #out = ln_p_β + ln_p_γ + ln_likelihood

    # ## prior only.
    # out = ln_p_β + ln_p_γ

    ## unnormalized likelihood only.
    #out = unnormalized_ln_likelihood

    # ## unnormalized posterior.
    out = ln_p_β + ln_p_γ + unnormalized_ln_likelihood

    return out
    #return out, μ
end

function solveSIRnumerical( I_0::T,
                            S_0::T,
                            R_0::T,
                            β::T,
                            γ::T,
                            N::T;
                            t0::T = 0.0,
                            t_end::T = 365.0) where T <: Real
    #
    θ = [N; β; γ]
    u0 = [S_0; I_0; R_0]

    tspan = (t0, t_end)
    prob = DifferentialEquations.ODEProblem(SIR!, u0, tspan, θ)
    sol = DifferentialEquations.solve(prob)

    S = tt->sol(tt)[1]
    I = tt->sol(tt)[2]
    R = tt->sol(tt)[3]

    return S, I, R
end

######

function SIR!(du, u, θ, t)

  susceptible = u[1]
  infected = u[2]
  removed = u[3]

  N = θ[1]
  β = θ[2]
  γ = θ[3]

  obj1 = susceptible*infected*β/N
  obj2 = γ*infected

  du[1] = -obj1
  du[2] = obj1 - obj2
  du[3] = obj2
end
