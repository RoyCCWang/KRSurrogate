

function runbatchtransport(src_dist, T_map, ds_T_map, dT;
                            N_visualization = 1000)

    ## visualize transport.
    X = collect( rand(src_dist) for n = 1:N_visualization )

    println("Timing: transport.")
    @time results = T_map.(X)
    println("end timing.")
    println()

    x_array = collect( results[n][1] for n = 1:N_visualization)
    discrepancy_array = collect( results[n][2] for n = 1:N_visualization)


    max_val, max_ind = findmax(norm.(discrepancy_array))
    println("l-2 norm( abs(u-u_rec) ), summed over all dimensions: ", sum(norm.(discrepancy_array)))
    println("largest l-1 discrepancy is ", max_val)
    println("At that case: x = ", x_array[max_ind])
    println()

    return x_array, discrepancy_array, X
end

# to do: stres test.
function verifytransportderivatives(src_dist, T_map, ds_T_map, dT, D)

    ##
    x0 = rand(src_dist)

    y0, err_y0 = T_map(x0)


    dT_ND = xx->FiniteDiff.finite_difference_jacobian(aa->T_map(aa)[1], xx)

    T_map_components = collect( xx->T_map(xx)[1][d] for d = 1:D )

    # popular numerical differentiation package.
    d2T_components_ND = collect( xx->FiniteDiff.finite_difference_hessian(T_map_components[d], xx) for d = 1:D )

    # legacy numerical differentiation package.
    #d2T_components_ND = collect( xx->Calculus.hessian(T_map_components[d], xx) for d = 1:D )

    # higher-order numerical differentiation.
    #d2T_components_ND1 = collect( xx->numericalhessian(T_map_components[d], xx) for d = 1:D )


    d2T_ND = xx->collect( d2T_components_ND[d](xx) for d = 1:D )

    ## test.

    println("test T")
    x0 = rand(src_dist)

    y0, err_y0 = T_map(x0)
    println("err_y0 = ", err_y0)
    println()

    println(" test ds_T_map ")
    dT_x0_AN, d2T_x0_AN = ds_T_map(x0,y0)

    dT_x0_ND = dT_ND(x0)
    println("ND: dT(x0)")
    display(dT_x0_ND)
    println()

    println("AN: dT(x0)")
    display(dT_x0_AN)
    println()

    println("dT discrepancy: ", norm(dT_x0_AN-dT_x0_ND))
    println()

    d2T_x0_ND = d2T_ND(x0)
    println("ND: d2T(x0)")
    display(d2T_x0_ND)
    println()

    println("AN: d2T(x0)")
    display(d2T_x0_AN)
    println()

    # println("d2T discrepancy ratio: ",
    #       evalKRd2Tratiodiscrepancy(d2T_x0_AN, d2T_x0_ND))
    # println()

    disc_ratio = evalKRd2Tratiodiscrepancy(d2T_x0_AN, d2T_x0_ND)
    println("d2T discrepancy ratio: ", disc_ratio)
    println()

    ratio_tol = 1e-2
    println("Are the ratios close to one, within a tol of ", ratio_tol, ", per dim?")
    println( ratiodiscrepancywithintol(disc_ratio; tol = ratio_tol) )


    ### again.
    println()
    println()
    println(" again ds_T_map ")
    x0 = rand(src_dist)

    y0, err_y0 = T_map(x0)
    dT_x0_AN, d2T_x0_AN = ds_T_map(x0,y0)

    dT_x0_ND = dT_ND(x0)
    println("ND: dT(x0)")
    display(dT_x0_ND)
    println()

    println("AN: dT(x0)")
    display(dT_x0_AN)
    println()

    println("dT discrepancy: ", norm(dT_x0_AN-dT_x0_ND))
    println()

    d2T_x0_ND = d2T_ND(x0)
    println("ND: d2T(x0)")
    display(d2T_x0_ND)
    println()

    println("AN: d2T(x0)")
    display(d2T_x0_AN)
    println()

    # println("d2T discrepancy ratio: ",
    #       evalKRd2Tratiodiscrepancy(d2T_x0_AN, d2T_x0_ND))
    # println()

    disc_ratio = evalKRd2Tratiodiscrepancy(d2T_x0_AN, d2T_x0_ND)
    println("d2T discrepancy ratio: ", disc_ratio)
    println()

    ratio_tol = 1e-2
    println("Are the ratios close to one, within a tol of ", ratio_tol, ", per dim?")
    println( ratiodiscrepancywithintol(disc_ratio; tol = ratio_tol) )

end
