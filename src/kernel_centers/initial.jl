
#### Schemes for coming up with the canididates.

function drawsamplinglocationsuniform(  N::Int,
                                        a::Vector{T},
                                        b::Vector{T})::Vector{Vector{T}} where T <: Real
    D = length(b)
    @assert length(a) == D

    X = Vector{Vector{T}}(undef,N)

    for n = 1:N
        X[n] = Vector{T}(undef,D)
        for d = 1:D
            X[n][d] = Utilities.convertcompactdomain(rand(), zero(T), one(T), a[d], b[d])
        end
    end

    return X
end


## f is an unnormalize probability density.
# truncation_factor * 10 is the percent of maximum f that is
#   deemed too low for keeping.
function getinitialcandidatesviauniform(  N::Int,
                                a::Vector{T},
                                b::Vector{T},
                                f::Function;
                                zero_tol::T = 1e-12,
                                truncation_factor = 0.1 ) where T <: Real
    D = length(b)
    @assert length(a) == D

    println("Drawing from uniform proposal.")
    @time X = drawsamplinglocationsuniform(N, a, b)

    println("Find max.")
    y0 = f.(X)
    @time max_y0 = maximum(y0)
    @assert max_y0 > zero_tol

    y = y0 ./ max_y0

    println("sortperm")
    @time ind = sortperm(y, rev = true)

    y = y[ind]
    X = X[ind]

    # prune the locations below a center threshold.
    println("findfirst")
    @time t_ind = findfirstsorteddescending(y, truncation_factor)

    if t_ind != 0
        y = y[1:t_ind]
        X = X[1:t_ind]
    end

    println("exit.")
    return X, y
end


#### select initial kernel centers.

"""
h: 𝓧 -> ℝ_{+} indicates how preferred it is for an input to be selected.
The larger the value, the more preferred.
f is the target density function for fitting.
Removes locations from X_candidates that are kernel centers.
"""
function selectkernelcenters!( X_candidates::Vector{Vector{T}},
                            additive_function::Function,
                            f::Function,
                            θ_kDPP_base::KT,
                            θ_RKHS::KT2;
                            max_iters_RKHS::Int = 5000,
                            base_gain::T = 1.0,
                            kDPP_zero_tol::T = 1e-12,
                            N_kDPP_draws::Int = 1,
                            N_kDPP_per_draw = 50,
                            zero_tol_RKHS::T = 1e-13,
                            prune_tol::T = 1.1*zero_tol_RKHS,
                            σ²::T = 1e-3) where {T,KT,KT2}
    #
    θ_kDPP = RKHSRegularization.AdditiveVarianceKernelType( θ_kDPP_base,
                            additive_function,
                            base_gain,
                            kDPP_zero_tol)
    #


    X_kDPP::Vector{Vector{T}} = Vector{Vector{T}}(undef, 0)
    𝑖_kDPP::Vector{Int} = Vector{Int}(undef, 0)

    if length(X_candidates) > N_kDPP_per_draw
        X_kDPP, 𝑖_kDPP = selectelementviaunionofkDPP(X_candidates,
                                    N_kDPP_draws,
                                    N_kDPP_per_draw,
                                    θ_kDPP)
    else
        println("X_candidates = ", length(X_candidates))
        X_kDPP = copy(X_candidates)
        𝑖_kDPP = collect( i for i = 1:length(X_kDPP))
    end

    # fit.
    X_fit = X_kDPP
    f_X_fit = f.(X_fit)
    c_q, 𝓧_q,
        keep_indicators = fitRKHSdensity(  f_X_fit,
                                X_fit, max_iters_RKHS, σ²,
                                θ_RKHS, zero_tol_RKHS,
                                prune_tol)

    # remove kernel centers from candidate pool.
    # I am here. need new index tracking since we included X0.
    ind_kernel_centers = sort(𝑖_kDPP[keep_indicators])
    deleteat!(X_candidates, ind_kernel_centers)

    return c_q, 𝓧_q, X_fit
end
