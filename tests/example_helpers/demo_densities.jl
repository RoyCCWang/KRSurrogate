
# case where we have the oracle target distribution in Distributions.dist form.
function preparesyntheticdist(Y_pdf, Y_dist, x_ranges)

    D = length(x_ranges)

    X_nD = Utilities.ranges2collection(x_ranges, Val(D))
    tmp = Y_pdf.(X_nD)
    f_scale_factor = maximum(tmp)

    f = xx->Y_pdf(xx)/f_scale_factor

    # prepare all marginal joint densities.
    f_joint = xx->evaljointpdf(xx,f,D)[1]

    return f, f_joint
end

function setupGMM3Dpub()
    D = 3
    N_GMM = 3

    #
    μ_array = Vector{Vector{Float64}}(undef, D)
    Σ_array = Vector{Matrix{Float64}}(undef, D)

    μ_array[1] = [2.4823906863445315; 3.3976346743245167; 3.530955629599614]
    Σ_array[1] = [0.4968429082264334 -0.004986315942663292 -0.19225541161486576;
        -0.004986315942663292 0.36940906360234965 -0.5638498787966733;
        -0.19225541161486576 -0.5638498787966733 1.0]

    #
    μ_array[2] = [8.767981140381082; 6.962138580303089; 6.684258394072378]
    Σ_array[2] = [1.0 -0.6856118413159787 0.5011711621185674;
        -0.6856118413159787 0.7636332896585137 0.008375203778496374;
        0.5011711621185674 0.008375203778496374 0.7122003375435665]
    #
    μ_array[3] = [9.156463071062559; 9.226928277753439; 7.888887358142144]
    Σ_array[3] = [0.2314699847422127 -0.026057067615439744 -0.07686680062396627;
        -0.026057067615439744 1.0 -0.26406767226317546;
        -0.07686680062396627 -0.26406767226317546 0.1257888087791072]

    # multivariate Cauchy via product of univariate Cauchy distributions.
    μ_cauchy = [ 1.0203583600735444; -1.6784792331930203; 0.38524474343877047]
    σ_cauchy = [  2.6355657198558275; 2.562250113788404; 2.754563266817158]

    cauchy_dists = collect( Distributions.Cauchy(μ_cauchy[d], σ_cauchy[d]) for d = 1:D )
    product_cauchy = Distributions.Product(cauchy_dists)

    # mixture distribution.
    dist_gen_array = collect( Distributions.MvNormal(μ_array[i], Σ_array[i]) for i = 1:N_GMM )
    dist_gen_array = [dist_gen_array; product_cauchy]

    weight_gmm = 0.75 # good for full fit.
    #weight_gmm = 0.5
    mixture_weights = [weight_gmm/3; weight_gmm/3; weight_gmm/3; 1-weight_gmm]

    dist_gen = Distributions.MixtureModel(dist_gen_array, mixture_weights)

    pdf = xx->Distributions.pdf(dist_gen,xx)

    return pdf, dist_gen
end

function setupskipDgmm3D(limit_a, limit_b, N_array)
    D = 3

    Y_pdf, Y_dist = setupGMM3Dpub()

    x_ranges = collect( LinRange(limit_a[d], limit_b[d], N_array[d]) for d = 1:D )

    return Y_pdf, Y_dist, x_ranges
end



function getk05helmetgrayscale(ch_select::Int = 1) where T <: Real
    #
    @assert 1 <= ch_select <= 4

    D = 2

    imA = PyPlot.imread("/home/roy/MEGAsync/data/image/kodak_helmet/k05_helmet_30.png")
    #imA = PyPlot.imread("/home/roy/MEGAsync/data/image/kodak_helmet/k05_helmet_12x12.png")
    #imA = PyPlot.imread("/home/roy/MEGAsync/data/image/kodak_helmet/wiggle_12x12.png")

    imB0 = Matrix{Float64}(undef,size(imA,1), size(imA,2))
    imB0[:,:] = imA[:,:,ch_select]

    #imB = abs( minimum(imB0) )*3.0 .+ imB0 # higher chance that f(x) is > 0, for all x.
    imB = abs( 1e-5 )*3.0 .+ imB0 # higher chance that f(x) is > 0, for all x.

    Nr, Nc = size(imB)
    x_ranges = collect( LinRange(1, size(imB,i), size(imB,i)) for i = 1:2 )


    samples = sqrt.(abs.(imB))
    f_sqrt, df_unused, d2f_unused = Utilities.setupcubicitp(samples, x_ranges, 1.0)

    X_nD = Utilities.ranges2collection(x_ranges, Val(D))
    scale_factor = maximum(f_sqrt.(X_nD))

    f = xx->(f_sqrt(xx)/scale_factor)^2

    ### get limits.
    limit_a = [1.0; 1.0]
    limit_b = [Nr; Nc] .* 1.0

    return f, x_ranges, limit_a, limit_b, fig_num
end
