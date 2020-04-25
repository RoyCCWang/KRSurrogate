
function visualizefit2Dmarginal(fig_num,
                                c_array,
                                𝓧_array,
                                θ_array,
                                limit_a,
                                limit_b,
                                f_joint::Function;
                                max_integral_evals = 10000,
                                N_visualize_2D_per_dim = 100,
                                N_visualize_1D = 300,
                                display_1D_flag = true,
                                Y_array = [],
                                X_array_debug = [],
                                f_X_array_debug = [] )
    #
    for d = 1:length(X_array_debug)
        @assert length(X_array_debug[d]) == length(f_X_array_debug[d])
    end

    gq_array, CDF_array = packagefitsolution(c_array, θ_array, 𝓧_array;
                            max_integral_evals = max_integral_evals)


    xv_ranges = collect( LinRange(limit_a[d], limit_b[d], N_visualize_2D_per_dim) for d = 1:2 )
    Xv_nD = Utilities.ranges2collection(xv_ranges, Val(2))

    println("Timing, f_joint.(Xv_nD)")

    @time f_Xv_nD = reshape(parallelevals(f_joint, vec(Xv_nD)), size(Xv_nD))
    println("End timing.")
    fig_num = VisualizationTools.visualizemeshgridpcolor(xv_ranges,
                      f_Xv_nD, [], "x", fig_num, "f, numerical integration")


    d_select = 2

    g2 = gq_array[d_select]
    g2_Xv_nD = g2.(Xv_nD)
    fig_num = VisualizationTools.visualizemeshgridpcolor(xv_ranges,
                      g2_Xv_nD, 𝓧_array[d_select], "x", fig_num,
                          "g2, markers at kernel centers")
    #

    if display_1D_flag
        PyPlot.figure(fig_num)
        fig_num += 1


        xq = LinRange(limit_a[1], limit_b[1], N_visualize_1D)
        Xq = collect( [xq[n]] for n = 1:length(xq) )

        fq_Xq = gq_array[1].(Xq)
        fq_𝓧 = gq_array[1].(𝓧_array[1])

        # this is slow.
        println("Timing, f(Xq)")
        #f_Xq = f_joint.(Xq)
        @time f_Xq = reshape(parallelevals(f_joint, vec(Xq)), size(Xq))

        PyPlot.plot(xq, fq_Xq, label = "fq")
        PyPlot.plot(𝓧_array[1], fq_𝓧, "x", label = "fq kernel centers")
        PyPlot.plot(xq, f_Xq, "--", label = "f")

        if !isempty(Y_array)
            PyPlot.plot(x_ranges[1], Y_array[1], "^", label = "Y")
        end

        if !isempty(X_array_debug)
            PyPlot.plot(X_array_debug[1], f_X_array_debug[1], "s", label = "samples used for fitting")
        end

        PyPlot.title("f vs. fq")
        PyPlot.legend()
    end

    return fig_num
end

function visualize2Dhistogram(  fig_num::Int,
                                x_array,
                                limit_a,
                                limit_b;
                                n_bins::Int = 500,
                                use_bounds = true,
                                axis_equal_flag = false,
                                title_string = "",
                                flip_vertical_flag = false)
    #
    N_visualization = length(x_array)

    bounds = [[limit_a[2], limit_b[2]], [limit_a[1], limit_b[1]]]

    #
    PyPlot.figure(fig_num)
    fig_num += 1
    p1 = collect(x_array[n][2] for n = 1:N_visualization)
    p2 = collect(x_array[n][1] for n = 1:N_visualization)

    if use_bounds
        PyPlot.plt.hist2d(p1, p2, n_bins, range = bounds, cmap="jet")
    else
        PyPlot.plt.hist2d(p1, p2, n_bins, cmap="jet" )
    end
    PyPlot.title("Visualizing the empirical marginal joint transported samples for the first two dimensions.")

    # PyPlot.figure(fig_num)
    # fig_num += 1
    # p1 = collect(randn() for n = 1:N_visualization)
    # p2 = collect(randn() for n = 1:N_visualization)
    #
    # if use_bounds
    #     PyPlot.plt.hist2d(p1, p2, n_bins, range = bounds, cmap="Greys")
    # else
    #     PyPlot.plt.hist2d(p1, p2, n_bins, cmap="Greys" )
    # end

    if axis_equal_flag
        PyPlot.plt.axis("equal")
    end

    PyPlot.title(title_string)
    PyPlot.colorbar()

    if flip_vertical_flag
        PyPlot.plt.gca().invert_yaxis()
    end

    return fig_num
end

function imshowgrayscale(fig_num::Int, A, title_string::String)
    PyPlot.figure(fig_num)
    fig_num += 1

    PyPlot.imshow(f.(Xp), cmap="gray")
    PyPlot.title(title_string)
    PyPlot.colorbar()

    return fig_num
end





function separatecomponents(Y::Vector{Vector{T}})::Vector{Vector{T}} where T

    D = length(Y[1])
    N = length(Y)

    out = Vector{Vector{T}}(undef, D)

    for d = 1:D
        out[d] = Vector{T}(undef, N)

        for n = 1:N

            out[d][n] = Y[n][d]
        end
    end

    return out
end



function plotallpairwisecontours(X_array::Vector{Vector{T}},
                                dummy_handle::HT,
                                N_bins::Int,
                                N_bins_h1::Int,
                                output_folder_name::String) where {T,HT}

    #
    #N = length(X_array[1])
    D = length(X_array)

    for d = 1:D
        handles_d = Vector{HT}(undef, d)

        # 2D contour density plot of pair-wise marginals.
        for k = 1:d-1
            # prepare samples for each dimension.
            x = X_array[d]
            y = X_array[k]

            # empirical density of pair-wise marginal dim d vs. dim k,
            #   (horizontal vs. vertical on plot).
            h = StatsBase.fit(StatsBase.Histogram, (x, y),
                        closed = :left, nbins = (N_bins, N_bins));

            handles_d[k] = Plots.contour(StatsBase.midpoints(h.edges[1]),
                                 StatsBase.midpoints(h.edges[2]),
                                 h.weights';
                                 colorbar = :none,
                                 #aspect_ratio = :equal,
                                 reuse = false)
            #


            save_fig_name = Printf.@sprintf("%s/d-k_%d-%d.png",
                                output_folder_name, d, k)
            Plots.savefig(handles_d[k], save_fig_name)
        end

        # plot.
        width_tuple = Tuple( 1/d for i = 1:d )
        height_tuple = Tuple( 1.0 for i = 1:d )
        fig_d = Plots.plot(handles_d..., layout = Plots.grid(1,d, heights=height_tuple, widths=width_tuple))


        # display.
        display(fig_d)
    end

    return nothing
end
