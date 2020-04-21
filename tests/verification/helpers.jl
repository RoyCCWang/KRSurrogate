

function preparefitsynthdensitydemo!(   a_array::Vector{T},
                                        σ_array::Vector{T},
                                        D::Int,
                                        N_X::Vector{Int},
                                        𝑋::Vector{Vector{T}},
                                        skip_flag::Bool,
                                        f_joint::Function ) where T

    D_fit = D
    if skip_flag
        D_fit = D-1

        resize!(a_array, D_fit)
        resize!(σ_array, D_fit)
    end

    @assert length(a_array) == D_fit
    @assert length(σ_array) == D_fit

    println("Computing f_X_array")
    X_array = collect( collect( 𝑋[n][1:d] for n = 1:N_X[d] ) for d = 1:D_fit )

    # try this out.
    X_array = collect( unique(X_array[d]) for d = 1:D )


    @time f_X_array = collect( f_joint.(X_array[d]) for d = 1:D_fit )
    println("end timing.")

    return X_array, f_X_array, D_fit
end


function fitdensityadaptivebp(  f::Function,
                                f_joint::Function,
                                𝑋::Vector{Vector{T}},
                                N_X::Vector{Int},
                                x_ranges,
                                fit_optim_config::FitDensityConfigType;
                                skip_flag::Bool = true,
                                a_array = 0.1 .* ones(T, length(N_X)),
                                σ_array = sqrt(1e-5) .* ones(T, length(N_X)),
                                amplification_factor = 1.0,
                                zero_tol_RKHS = 1e-13,
                                prune_tol = 1.1*zero_tol_RKHS,
                                max_iters_RKHS = 10000,
                                max_integral_evals = 10000,
                                attenuation_factor_at_cut_off = 2.0,
                                N_bands = 5 ) where T
    #
    D = length(N_X)

    X_array, f_X_array, D_fit = preparefitsynthdensitydemo!(   a_array,
                                            σ_array, D, N_X, 𝑋,
                                            skip_flag, f_joint)

    # initial parameters.
    #zero_tol_RKHS = 1e-13
    #prune_tol = 1.1*zero_tol_RKHS
    #max_iters_RKHS = 10000
    #σ_array = sqrt(1e-5) .* ones(T, D_fit)
    #max_integral_evals = 10000
    #attenuation_factor_at_cut_off = 2.0
    #N_bands = 5

    # to do: Bayesian optimization here.
    println("Timing: fitproxydensities")
    @time c_array, 𝓧_array, θ_array,
            dφ_array, d2φ_array,
            dϕ_array, d2ϕ_array,
            Y_array = fitmarginalsadaptive(fit_optim_config, f_X_array, X_array,
                                        x_ranges[1:D_fit], f_joint,
                                        max_iters_RKHS, a_array, σ_array,
                                        amplification_factor, N_bands,
                                        attenuation_factor_at_cut_off,
                                        zero_tol_RKHS, prune_tol,
                                        max_integral_evals)

    #
    return c_array, 𝓧_array, θ_array,
            dφ_array, d2φ_array,
            dϕ_array, d2ϕ_array, Y_array
end


function fitdensitynonadaptive(      f::Function,
                                f_joint::Function,
                                𝑋::Vector{Vector{T}},
                                N_X::Vector{Int};
                                skip_flag::Bool = false,
                                a_array = 0.1 .* ones(T, length(N_X)),
                                σ_array = sqrt(1e-5) .* ones(T, length(N_X)),
                                zero_tol_RKHS = 1e-13,
                                prune_tol = 1.1*zero_tol_RKHS,
                                max_iters_RKHS = 10000,
                                max_integral_evals = 10000 ) where T
    #
    D = length(N_X)

    X_array, f_X_array, D_fit = preparefitsynthdensitydemo!(   a_array,
                                            σ_array, D, N_X, 𝑋,
                                            skip_flag, f_joint)

    # to do: Bayesian optimization here.
    println("Timing: fitskiplastnonadaptive")
    @time c_array, 𝓧_array,
        θ_array = fitskiplastnonadaptive(f_X_array, X_array,
                                        max_iters_RKHS, a_array, σ_array,
                                        zero_tol_RKHS, prune_tol)

    #
    return c_array, 𝓧_array, θ_array
end

function setupquantilefordemo(  f_target::Function,
                                x_ranges,
                                limit_a,
                                limit_b::Vector{T},
                                μ::Vector{T},
                                σ_array::Vector{T};
                                max_integral_evals = 10000,
                                quantile_iters = 1000,
                                N_nodes_tanh = 200,
                                m_tsq = 7,
                                quantile_err_tol = 1e-3, #0.01  both gave l-2 norm( abs(u-u_rec) ), summed over all dimensions: 0.024788099072235982
                                max_traversals = 50,
                                N_predictive_traversals = 40,
                                correction_epoch = 10,
                                quantile_max_iters = 500,
                                quantile_convergence_zero_tol = 1e-6, #1e-4
                                n_limit = 10,
                                n0 = 5) where T

    # quantile-related parameters.


    KR_config = KRTaylorConfigType( x_ranges,
                                    quantile_iters,
                                    max_integral_evals,
                                    N_nodes_tanh,
                                    m_tsq,
                                    quantile_err_tol,
                                    max_traversals,
                                    N_predictive_traversals,
                                    correction_epoch,
                                    quantile_max_iters,
                                    quantile_convergence_zero_tol,
                                    n_limit,
                                    n0 )

    #
    # set up source distribution.
    D = length(limit_a)

    src_dist = Distributions.MvNormal(μ, diagm(σ_array))

    ## bundled map.
    df_target = xx->ForwardDiff.gradient(f_target, xx)
    d2f_target = xx->ForwardDiff.hessian(f_target, xx)

    # force compile once.
    y_test = (limit_a + limit_b) ./ 2
    df_target(y_test)
    d2f_target(y_test)

    return src_dist, df_target, d2f_target, KR_config
end
