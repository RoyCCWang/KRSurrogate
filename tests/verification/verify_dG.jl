# using Liebniz.



fq = gq_array[2]

x_full = x0



df = xx->Calculus.derivative(fq, xx)

df_dx_eval_ND = df(x_full)
df_dx_eval_AN = collect( ∂f_∂x_array[d](x_full[end]) for d = 1:D )

println("df_dx_eval_ND = ", df_dx_eval_ND)
println("df_dx_eval_AN = ", df_dx_eval_AN)
println()

function evalhtilde(f, x_full, limit_a, limit_b, d)
    v = x_full[1:d-1]
    x = x_full[d]
    f_v = xx->f([v; xx])
    out = evalintegral(  f_v, limit_a[d], x_full[d])

    return out
end

function evalh(f, x_full, limit_a, limit_b, d)
    v = x_full[1:d-1]
    x = x_full[d]
    f_v = xx->f([v; xx])
    out = evalintegral(  f_v, limit_a[d], x_full[d])

    Z = evalintegral(f_v, limit_a[d], limit_b[d])
    return out/Z
end

x = x_full[d_select]
v = x_full[1:d_select-1]
a = limit_a[d_select]
b = limit_b[d_select]
f_v = xx->fq([v; xx])

h_tilde = xx->evalhtilde(fq, xx, limit_a, limit_b, d_select)
LHS = Calculus.gradient(h_tilde, x_full)

#∂f_loadx_array[1](x0)
RHS2 = evalintegral(∂f_∂x_array[1], a, x)

df1_v = xx->df([v;xx])[1]
RHS = evalintegral(df1_v, a, x)

println("LHS = ", LHS)
println("RHS = ", RHS)
println("RHS2 = ", RHS2)
println()

h = xx->evalh(fq, xx, limit_a, limit_b, d_select)
LHS = Calculus.gradient(h, x_full)

df1_v = xx->df([v;xx])[1]
h_x = evalintegral(f_v, a, x)
#Z_v = evalintegral(f_v, a, b)
∂h_x = evalintegral(∂f_∂x_array[1], a, x)
∂Z_v = evalintegral(∂f_∂x_array[1], a, b)
# ∂h_x = evalintegral(df1_v, a, x)
# ∂Z_v = evalintegral(df1_v, a, b)

numerator = ∂h_x*Z_v - h_x*∂Z_v
denominator = Z_v^2
RHS = numerator/denominator

println("LHS = ", LHS)
println("RHS = ", RHS)
println()
