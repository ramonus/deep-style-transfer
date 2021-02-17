using Flux
using Images
using CUDA
using Zygote
using Metalhead
using Statistics
using LinearAlgebra
using Plots
using Parameters:@with_kw

if has_cuda()
    @info "CUDA is on"
    device = gpu
else
    @info "CUDA is off"
    device = cpu
end

arrOcarr = Union{Array{T,n},CuArray{T,n}} where {T <: Float32,n}

mutable struct Model
    style1::Chain
    style2::Chain
    content1::Chain
end

struct ResLayers
    style1::arrOcarr{Float32,4}
    style2::arrOcarr{Float32,4}
    content::arrOcarr{Float32,4}
end

@with_kw struct StyleMatcher
    result::ResLayers
    chain1::Chain
    chain2::Chain
    function StyleMatcher(result::ResLayers)
        c1 = Chain(
            Conv(reshape(result.style1, size(result.style1)[begin:end - 1]), Flux.Zeros()),
        )
        c2 = Chain(
            Conv(reshape(result.style2, size(result.style2)[begin:end - 1]), Flux.Zeros())
        )
        new(result, c1, c2)
    end
end

Flux.@functor Model

function model()
    vgg = VGG19().layers
    style1 = Chain(vgg[1:7]...)
    style2 = Chain(vgg[8:12]...)
    content1 = Chain(vgg[13:13]...)
    Model(style1, style2, content1)
end

function (m::Model)(target_image::Array{Float32,4})
    s1 = m.style1(target_image)
    s2 = m.style2(s1)
    c1 = m.content1(s2)
    ResLayers(s1, s2, c1)
end

function img_to_arr(img::Array{<:AbstractRGB,2}, dsize=nothing)
    if dsize !== nothing
        img = imresize(copy(img), dsize)
    end

    cat(red.(img), green.(img), blue.(img), dims=3)
end
function preprocess_array(arr::Array{<:Real,3})::Array{Float32,4}
    narr = Float32.(arr)
    narr = 2 .* narr .- 1
    Flux.unsqueeze(narr, 4)
end
function deprocess_array(arr::Array{Float32,4})::Array{Float32,3}
    narr = (arr[:,:,:,1] .+ 1) ./ 2
    map(clamp01nan, narr)
end
function arr_to_img(arr::Array{Float32})::Array{RGB{Float32},2}
    RGB.(arr[:,:,1], arr[:,:,2], arr[:,:,3])
end

function style_loss(rcurrent::ResLayers, rtarget::ResLayers)
    loss = 0f0
    sm = StyleMatcher(rtarget)
    return sm
    results = Any[]
    for i in 1:length(current_style)
        cs = current_style[i]
        ts = target_style[i]
        m = chains[i]
        res = m(cs)
        println("SIZE $i: ", size(res))
    end
end
function content_loss(rcurrent::ResLayers, rtarget::ResLayers)::Float32
    sum((rcurrent.content - rtarget.content).^2)
end

function test(rng)
    style = Images.load("img.jpg")
    content = Images.load("img1.jpg")
    sarr, carr = map(preprocess_array, img_to_arr.((style, content), ((250, 250),)))
    
    tarr = 2 .* rand(Float32, size(carr)) .- 1

    m = model()
    rstyle = m(sarr)
    # (cont_1, _, _) = m(carr)
    rtar = m(tarr)
    l = style_loss(rtar, rstyle)
    rest = l.chain1(reshape(rtar.style1, size(rtar.style1)[begin:end - 1]))[1,:,:]
    println(size(rest))
    argm = argmax(rest, dims=1)
    println("Size argm: ", size(argm))
    println(argm[1:5])
    final = nothing
    for i = rng
        ind = argm[i]
        maxval = rest[ind]
        t = rtar.style1[:,:,i]
        s = rstyle.style1[:,:,ind[1]]
        println("R($i): ", sum(t .* s), " ", extrema(rest[:,i]), " ", maxval)
        println("Targ: ", extrema(t))
        println("Sty: ", extrema(s))
        r = hcat(t, s)
        if final === nothing
            final = r
        else
            final = vcat(final, r)
        end
    end
    final
end