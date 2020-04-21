
struct KernelCenterConfigType{T}
    amplification_factor::T
    attenuation_factor_at_cut_off::T
    N_bands::Int
    N_preliminary_candidates::Int
    candidate_truncation_factor::T
    candidate_zero_tol::T
    base_gain::T
    kDPP_zero_tol::T
    N_kDPP_draws::Int
    N_kDPP_per_draw::Int
    zero_tol_RKHS::T
    prune_tol::T
    close_radius_tol::T
    N_refinements::Int
    max_iters_RKHS::Int
    σ²::T
    initial_divisions::Int
end

struct KRTaylorConfigType{T,RT}
    x_ranges::Vector{RT}
    quantile_iters::Int
    max_integral_evals::Int
    N_nodes_tanh::Int
    m_tsq::Int
    quantile_err_tol::T
    max_traversals::Int
    N_predictive_traversals::Int
    correction_epoch::Int
    quantile_max_iters::Int
    quantile_convergence_zero_tol::T
    n_limit::Int
    n0::Int
end

abstract type KRDomainType end

struct FiniteDomainType <: KRDomainType
end


#### config parameters for fitting densities.

abstract type FitDensityConfigType end

struct SDPConfigType{T} <: FitDensityConfigType
    zero_tol::T
    max_iters::Int
end

struct ROptimConfigType{T} <: FitDensityConfigType
    zero_tol::T
    max_iters::Int

    verbose_flag::Bool
    max_iter_tCG::Int
    ρ_lower_acceptance::T # recommended to be less than 0.25
    ρ_upper_acceptance::T
    minimum_TR_radius::T
    maximum_TR_radius::T
    norm_df_tol::T
    objective_tol::T
    avg_Δf_tol::T # zero if what to run for max_iters.
    avg_Δf_window::Int # the number of iterations to avg over.
    max_idle_update_count::Int  # the number of consecutive non-updates before declaring we're stuck.
    g::Function # the metric tensor is g(x) * I, g: ℝ_+^D → ℝ_+
    𝑟::T # the numerical delta used in approximating the Hessian, if Hessian is not posdef.
end

# a default constructor.
function ROptimConfigType(zero_tol::T, max_iters::Int)::ROptimConfigType{T} where T

    verbose_flag::Bool = false
    max_iter_tCG = 100
    ρ_lower_acceptance = 0.2 # recommended to be less than 0.25
    ρ_upper_acceptance = 5.0
    minimum_TR_radius::T = 1e-3
    maximum_TR_radius::T = 10.0
    norm_df_tol = 1e-5
    objective_tol = 1e-5
    avg_Δf_tol = 0.0 #1e-12 #1e-5
    avg_Δf_window = 10
    max_idle_update_count = 50
    g::Function = pp->1.0/(dot(pp,pp)+1.0)
    𝑟 = 1e-2

    return ROptimConfigType(    zero_tol,
                                max_iters,
                                verbose_flag,
                                max_iter_tCG,
                                ρ_lower_acceptance,
                                ρ_upper_acceptance,
                                minimum_TR_radius,
                                maximum_TR_radius,
                                norm_df_tol,
                                objective_tol,
                                avg_Δf_tol,
                                avg_Δf_window,
                                max_idle_update_count,
                                g,
                                𝑟)
end
