# explore sum of RQ kernels as CDF.

using FFTW
import PyPlot
import BSON
import Optim
import Random
using LinearAlgebra

import Interpolations

PyPlot.close("all")
fig_num = 1

PyPlot.matplotlib["rcParams"][:update](["font.size" => 22, "font.family" => "serif"])

Random.seed!(25)

function evalg(x::T, w::Vector{T}, zs::Vector{T}, bs::Vector{T})::T where T <: Real

    out = zero(T)
    for i = 1:length(zs)
        τ = x-zs[i]
        out += w[i]*τ/sqrt(bs[i] + τ^2)
    end

    return out
end

function evalZ(α::T, β::T, w::Vector{T}, zs::Vector{T}, bs::Vector{T})::T where T <: Real

    return evalg(β, w, zs, bs) - evalg(α, w, zs, bs)
end

function evalB(α::T, w::Vector{T}, zs::Vector{T}, bs::Vector{T}, Z::T)::T where T <: Real

    return evalg(α, w, zs, bs)/Z
end

function evalCDF(x::T, B::T, Z::T, w::Vector{T}, zs::Vector{T}, bs::Vector{T})::T where T <: Real
    return evalg(x, w, zs, bs)/Z - B
end

function evalPDF(x::T, w::Vector{T}, zs::Vector{T}, bs::Vector{T}, Z)::T where T <: Real

    return sum( w[n]*bs[n]/sqrt(bs[n]+(x-zs[n])^2 )^3 for n = 1:length(bs) )/Z
end

function convertcompactdomain(x::T, a::T, b::T, c::T, d::T)::T where T <: Real

    return (x-a)*(d-c)/(b-a)+c
end


lb = -3.0
ub = 4.0

L = 5
zs = collect( convertcompactdomain(rand(), 0.0, 1.0, lb, ub) for l = 1:L )
bs = rand(L) .* 5.0
w = rand(L) .* 7.0

Z = evalZ(lb, ub, w, zs, bs)
B = evalB(lb, w, zs, bs, Z)
CDF_func = xx->evalCDF(xx, B, Z, w, zs, bs)
PDF_func = xx->evalPDF(xx, w, zs, bs, Z)


### visualize.

x = LinRange(lb, ub, 500)

PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(x, CDF_func.(x))


PyPlot.legend()
PyPlot.xlabel("x")
PyPlot.ylabel("Q inv")
PyPlot.title("CDF")


PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(x, PDF_func.(x))


PyPlot.legend()
PyPlot.xlabel("x")
PyPlot.ylabel("p(x)")
PyPlot.title("PDF")

###### NiLang goes here.

@assert 1==2

@i function evalg2(out, x::T, w::Vector{T}, zs::Vector{T}, bs::Vector{T}) where T
	@routine begin
		buffer ← zeros(T, length(w))

		τs ← zeros(T, length(zs))
		numerators ← zeros(T, length(zs))
		denominators ← zeros(T, length(zs))

		τ_sqs ← zeros(T, length(zs))
		blahs ← zeros(T, length(zs))
		τ_denominators ← zeros(T, length(zs))

	    for i = 1:length(zs)

			τs[i] += x-zs[i]
			numerators[i] += w[i]*τs[i]

			τ_denominators[i] += x-zs[i]
			τ_sqs[i] += τ_denominators[i]^2

			blahs[i] += bs[i] + τ_sqs[i]
			denominators[i] += blahs[i]^0.5

	        buffer[i] += numerators[i]/denominators[i]
		end
    end

	out += sum(buffer)

    ~@routine
end

evalg2(0.0, x[32], w, zs, bs)
evalg(x[32], w, zs, bs)


### check inverse.

# generate an y value, given an arb. known oracle x value.
oracle_x = -0.34
y = evalg(oracle_x, w, zs, bs) # 4.543172925880954,

(~evalg2)(y, w, zs, bs)

### conclusion: reversible computing is a way to write bijective prorams, not find inverses.
# https://github.com/GiggleLiu/NiLang.jl/issues/34
# i.e., given output to a program that mutate inputs via reversible operations, get the inputs.
## perhaps not immediately useful to do inverses, but useful to "re-render" intermediate paths/states given final simulation.

### sample from NiLang tutorial. https://giggleliu.github.io/NiLang.jl/dev/notebooks/basic.html
@i function power1000(out, x::T) where T
	@routine begin
		xs ← zeros(T, 1000)
		xs[1] += 1
		for i=2:1000
			xs[i] += xs[i-1] * x
		end
	end

	out += xs[1000] * x # the final answer, the 1000-th multiplication with x.

	~@routine
end

power1000(0.0, 1.001)
(~power1000)(2.71692, 1.001)
