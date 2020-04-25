
function setupsyntheticdataset( N_components::Int,
                                N_array::Vector{Int},
                                x_a::Vector{T},
                                x_b::Vector{T},
                                setup_func::Function,
                                ::Val{D}) where {T,D}
    #
    @assert D == length(x_a) == length(x_b)
    for d = 1:D
        @assert x_a[d] < x_b[d]
    end

    mixture_weights = ones(Float64,N_components)./N_components

    pdf, Y_dist = setup_func(mixture_weights, D)

    # specify sampling positions for warp map.

    x_ranges = collect( LinRange(x_a[d], x_b[d], N_array[d]) for d = 1:D )
    #X_nD = Utilities.ranges2collection(x_ranges, Val(D))

    return Y_dist, pdf, x_ranges#, X_nD
end

# function myfunc(   x_ranges::Vector{LinRange{T}},
#                                     Y::Matrix{T},
#                                     marker_locations::Vector,
#                                     marker_symbol::String,
#                                     fig_num::Int,
#                                     title_string::String,
#                                     x1_title_string::String = "Dimension 1",
#                                     x2_title_string::String = "Dimension 2") where T <: Real
#     #
#     @assert length(x_ranges) == 2
#     x_coords = collect( collect(x_ranges[d]) for d = 1:2 )
#
#     PyPlot.figure(fig_num)
#     fig_num += 1
#     PyPlot.pcolormesh(x_coords[2], x_coords[1], Y, cmap = "Greys_r")
#     PyPlot.xlabel(x2_title_string)
#     PyPlot.ylabel(x1_title_string)
#     PyPlot.title(title_string)
#
#     for i = 1:length(marker_locations)
#         pt = reverse(marker_locations[i])
#         PyPlot.annotate(marker_symbol, xy=pt, xycoords="data")
#     end
#
#     PyPlot.plt.colorbar()
#
#     return fig_num
# end
#
function getk05helmetgrayscale( fig_num::Int, ch_select::Int,
                                    N_array_factor::T) where T <: Real
    #
    @assert 1 <= ch_select <= 4

    D = 2


    imA = PyPlot.imread("../data/k05_helmet_30.png")
    imB = Matrix{T}(undef,size(imA,1), size(imA,2))

    imB[:,:] = imA[:,:,ch_select]

    # visualize.
    PyPlot.figure(fig_num)
    fig_num += 1

    PyPlot.imshow(imB, cmap="gray")

    title_string = Printf.@sprintf("image at channel %d", ch_select)
    PyPlot.title("title_string")

    ### get interpolation function.
    imC = Matrix{T}(undef, size(imB,1), size(imB,2))
    for i = 1:size(imC,1)
        for j = 1:size(imC,2)
            imC[i,j] = imB[end-i+1,j]
        end
    end
    itp_f = Interpolations.interpolate(imC,
                Interpolations.BSpline(Interpolations.Quadratic(
                    Interpolations.Flat(Interpolations.OnGrid()))))

    etp_f = Interpolations.extrapolate(itp_f, Interpolations.Line())
    f = xx->abs(itp_f(xx...))

    ### get limits.
    limit_a = [1.0; 1.0]
    limit_b = [size(imC,1), size(imC,2)] .* 1.0
    N_array = round.(Int, [size(imC,1); size(imC,2)] .* N_array_factor )
    x_ranges = collect( LinRange(limit_a[d], limit_b[d], N_array[d]) for d = 1:D )

    return f, x_ranges, limit_a, limit_b, fig_num
end


# a is the shape parameter. θ is the scle parameter.
# Does not check if R is positive-definite.
function evalGaussiancopula(  x,
                            R::Matrix{T},
                            marginal_dists) where T <: Real
    D = length(x)

    PDF_eval_x = collect( Distributions.pdf(marginal_dists[d], x[d]) for d = 1:D )
    probit_CDF_eval_x = collect( probit( Distributions.cdf(marginal_dists[d], x[d]) ) for d = 1:D )
    CDF_eval_x = collect( Distributions.cdf(marginal_dists[d], x[d]) for d = 1:D )

    out = Gaussiancopula(probit_CDF_eval_x,R)*prod(PDF_eval_x)

    if isfinite(out) != true
        println(prod(PDF_eval_x))
        println(probit_CDF_eval_x)
        println(CDF_eval_x)
        println(x)
    end
    @assert isfinite(out) == true
    return out
end

function Gaussiancopula(u,Σ::Matrix{T}) where T
    N=length(u)
    @assert size(Σ)==(N,N)
    I_mat = LinearAlgebra.Diagonal{T}(LinearAlgebra.I, N)

    if all(isfinite.(u))
        return exp(-0.5*LinearAlgebra.dot(u,(inv(Σ)-I_mat)*u))/sqrt(LinearAlgebra.det(Σ))
    end

    return zero(T)
end

# inverse CDF of standard univariate normal
function probit(p::T)::T where T <: Real
    #@assert zero(T) <= p <= one(T)
    p = clamp(p,zero(T),one(T))

    return sqrt(2)*SpecialFunctions.erfinv(2*p-one(T))
end

function getmixture2Dbetacopula1( τ::T,
                                    N_array::Vector{Int}) where T
    #

    D = 2
    limit_a = [τ; τ]
    limit_b = [1-τ; 1-τ]

    x_ranges = collect( LinRange(limit_a[d], limit_b[d], N_array[d]) for d = 1:D )

    R_g = [   1.12  -0.24;
        -0.24  0.07]
    R_f = [   0.61829  0.674525;
        0.674525  0.759165]

    a_g = [1.2; 3.4]
    θ_g = [2.0; 3.0]
    gamma_dists = collect( Distributions.Beta(a_g[d], θ_g[d]) for d = 1:D )

    g = xx->evalGaussiancopula(xx, R_g, gamma_dists)

    M = 5
    a = Vector{Vector{Float64}}(undef, M)
    a[1] = [1.2, 3.4]
    a[2] = [8.137763709838943, 8.773653512648599]
    a[3] = [10.778052251461023, 7.174287054713922]
    a[4] = [4.4419645076864525, 14.540862450228454]
    a[5] = [6.582089073364336, 8.060857932040115]

    θ = Vector{Vector{Float64}}(undef, M)
    θ[1] = [2.0, 3.0]
    θ[2] = [7.9914398321997275, 7.804725310098352]
    θ[3] = [2.5364332327991757, 14.67358895901066]
    θ[4] = [0.6668795542283168, 8.625209709368946]
    θ[5] = [2.6212936024181754, 12.445626221379737]

    #γ = 1.4
    #mix_weights = stickyHDPHMM.drawDP(γ, M)
    mix_weights = [0.13680543732370892;
                     0.0019470978511141439;
                     0.8395383241234364;
                     0.018327175970185964;
                     0.003381964731554627 ]

    mix_dists = collect( Distributions.MixtureModel(Distributions.Beta[
    Distributions.Beta(a[1][d], θ[1][d]),
    Distributions.Beta(a[2][d], θ[2][d]),
    Distributions.Beta(a[3][d], θ[3][d]),
    Distributions.Beta(a[4][d], θ[4][d]),
    Distributions.Beta(a[5][d], θ[5][d])], mix_weights) for d = 1:D )

    f = xx->evalGaussiancopula(xx, R_f, mix_dists)

    #
    h = xx->(0.15*f(xx)+0.85*g(xx))

    return h, x_ranges
end

function getnDGMMrealizations(   N_realizations::Int,
                                mixture_weights::Vector{T},
                                D::Int) where T <: Real

    pdf, dist_gen = getnDGMMrealizations(mixture_weights, D)

    ### generate observations.
    Y = collect( rand(dist_gen) for n = 1:N_realizations )

    return Y, pdf, dist_gen
end




function getnDGMMrealizations(  mixture_weights::Vector{T},
                                D::Int) where T <: Real
    #
    N_components = length(mixture_weights)

    m_gen_array = Vector{Vector{Float64}}(undef,N_components)
    Σ_gen_array = Vector{Matrix{Float64}}(undef,N_components)
    for i = 1:N_components
        m_gen_array[i] = 3*i .*ones(T,D)

        S = randn(T,D,D)
        #S = Matrix{Float64}(LinearAlgebra.I,D,D)
        Σ_gen_array[i] = S'*S
        Σ_gen_array[i] = Σ_gen_array[i]./maximum(Σ_gen_array[i])
    end

    return preparegmmdist(m_gen_array, Σ_gen_array, mixture_weights)
end

function preparegmmdist(μ_array::Vector{Vector{T}},
                        Σ_array::Vector{Matrix{T}},
                        mixture_weights::Vector{T}) where T <: Real
    #
    N_components = length(mixture_weights)

    dist_gen_array = collect( Distributions.MvNormal(μ_array[i], Σ_array[i]) for i = 1:N_components )
    dist_gen = Distributions.MixtureModel(dist_gen_array, mixture_weights)

    pdf = xx->sum( mixture_weights[n]*Distributions.pdf(dist_gen_array[n], xx) for n = 1:N_components )

    return pdf, dist_gen
end

# ### mixture of tuncated beta-marginal Gaussian copula and gamma-marginal Gaussian copula.
# function generaterandombetacopula( τ::T,
#                                     N_array::Vector{Int};
#                                     gamma_a_multiplier::T = 5.0,
#                                     gamma_θ_multiplier::T = 5.0,
#                                     beta_a_multiplier::T = 5.0,
#                                     beta_θ_multiplier::T = 5.0,
#                                     N_components = 5) where T
#     #
#
#     D = length(N_array)
#     limit_a = [τ; τ]
#     limit_b = [1-τ; 1-τ]
#
#     x_ranges = collect( LinRange(limit_a[d], limit_b[d], N_array[d]) for d = 1:D )
#
#     # gamma distribution.
#     R_g = Utilities.generaterandomposdefmat(D)
#     R_f = Utilities.generaterandomposdefmat(D)
#
#     a_g = abs.(randn(D)) .* gamma_a_multiplier
#     θ_g = abs.(randn(D)) .* gamma_θ_multiplier
#     gamma_dists = collect( Distributions.Beta(a_g[d], θ_g[d]) for d = 1:D )
#
#     g = xx->evalGaussiancopula(xx, R_g, gamma_dists)
#
#     #
#     a = collect( rand(D) .* beta_a_multiplier for m = 1:N_components )
#     θ = collect( rand(D) .* beta_θ_multiplier for m = 1:N_components )
#
#     beta_dists = collect( Distributions.Beta(a[m][d]) for d = 1:D, m = 1:N_components )
#
#     # assemble mixture.
#     mix_weights = rand(N_components)
#     mix_weights = mix_weights ./ sum(mix_weights)
#
#     mix_dists = collect( Distributions.MixtureModel(Distributions.Beta[
#                     beta_dists[d,:]...], mix_weights) for d = 1:D )
#
#     f = xx->evalGaussiancopula(xx, R_f, mix_dists)
#
#     #
#     h = xx->(0.05*f(xx)+0.95*g(xx))
#
#     return h, x_ranges, f, mix_dists, R_f, g, gamma_dists, R_g
# end

function generaterandombetacopula( τ::T,
                                    N_array::Vector{Int};
                                    gamma_a_multiplier::T = 2.0,
                                    gamma_θ_multiplier::T = 2.0,
                                    beta_a_multiplier::T = 2.0,
                                    beta_θ_multiplier::T = 2.0) where T
    #

    D = length(N_array)
    limit_a = ones(T, D) .* τ
    limit_b = ones(T, D) .* (1-τ)

    x_ranges = collect( LinRange(limit_a[d], limit_b[d], N_array[d]) for d = 1:D )

    # gamma distribution.
    R_g = Utilities.generaterandomposdefmat(D)
    R_f = Utilities.generaterandomposdefmat(D)

    a_g = abs.(randn(D)) .* gamma_a_multiplier
    θ_g = abs.(randn(D)) .* gamma_θ_multiplier
    gamma_dists = collect( Distributions.Beta(a_g[d], θ_g[d]) for d = 1:D )

    g = xx->evalGaussiancopula(xx, R_g, gamma_dists)

    #
    a_b = rand(D) .* beta_a_multiplier
    θ_b = rand(D) .* beta_θ_multiplier

    beta_dists = collect( Distributions.Beta(a_b[d], θ_b[d]) for d = 1:D )


    f = xx->evalGaussiancopula(xx, R_f, beta_dists)

    # generate mixture weights.
    w_g = rand()*0.2
    w_b = rand()*0.8
    h = xx->(w_b*f(xx)+w_g*g(xx))

    return h, x_ranges, f, beta_dists, R_f, w_b,
                        g, gamma_dists, R_g, w_g

end


function inverseprobit(x::T)::T where T <: Real
    return (one(T)-SpecialFunctions.erf(-x/sqrt(2)))/2
end

function drawfromGaussiancopula(inv_cdfs::Vector,
                                R::Matrix{T}) where T <: Real
    #
    D = size(R)[1]

    L = cholesky(R).L
    z = collect( randn() for d = 1:D )
    x = L*z
    for i = 1:D
        x[i] = inv_cdfs[i](inverseprobit(x[i]))
    end

    return x
end

# draws N samples of the specified copula.
function drawfromdemocopula(N::Int,
                            β_dists,
                            R_β::Matrix{T},
                            w_β::T,
                            γ_dists,
                            R_γ::Matrix{T},
                            w_γ::T) where T
    # set up.
    quantiles_γ = collect( qq->Distributions.quantile(γ_dists[d], qq) for d = 1:D )
    quantiles_β = collect( qq->Distributions.quantile(β_dists[d], qq) for d = 1:D )


    Y = Vector{Vector{T}}(undef, N)

    for n = 1:N

        if rand() < w_γ
            Y[n] = drawfromGaussiancopula(quantiles_γ, R_γ)
        else
            Y[n] = drawfromGaussiancopula(quantiles_β, R_β)
        end
    end

    return Y
end
