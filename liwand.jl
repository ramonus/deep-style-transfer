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

function patches_sampling(image::AbstractArray{Float32,4}, patch_size::Integer=3, stride::Integer=1)
    h, w = size(image)[1:2]
    patches = []
    for i in 1:stride:h - patch_size + 1 , j in 1:stride:w - patch_size + 1
        push!(patches, image[i:i + patch_size - 1,j:j + patch_size - 1,:,:])
    end
    cat(patches..., dims=4)
end

function patches_norm(patches::AbstractArray{Float32,4})
    norms = similar(patches, size(patches, 4))
    for i = 1:size(patches, 4)
        p = patches[:,:,:,i]
        norms[i] = sqrt.(sum(p.^2))
    end
    norms
end

@with_kw mutable struct Model
    style1::Chain
    style2::Chain
    content::Chain
    s1_loss::Union{Chain,Nothing} = nothing
    s2_loss::Union{Chain,Nothing} = nothing
end

@with_kw mutable struct ResLayers
    input::AbstractArray{Float32,4}
    style1::AbstractArray{Float32,4}
    style2::AbstractArray{Float32,4}
    s1_params::AbstractArray{Float32,4} = patches_sampling(style1)
    s2_params::AbstractArray{Float32,4} = patches_sampling(style2)
    content::AbstractArray{Float32,4}
    s1loss::Union{Chain,Nothing} = nothing
    s2loss::Union{Chain,Nothing} = nothing
    function ResLayers(args...;create_chain::Bool)
        if create_chain
            p1 = patches_sampling(style1)
            c1 = Chain(Conv(p1, Flux.Zeros()))
            p2 = patches_sampling(style2)
            c2 = Chain(Conv(p2, Flux.Zeros()))
            ResLayers(args..., s1_params=p1, s2_params=s2, s1loss=c1, s2loss=c2)
        else
            return nothing
        end
    end
end

Flux.@functor Model

function model()
    vgg = VGG19().layers
    style1 = Chain(vgg[1:7]...)
    style2 = Chain(vgg[8:12]...)
    content = Chain(vgg[13:13]...)
    Model(style1, style2, content)
end

function (m::Model)(target_image::Array{Float32,4})
    s1 = m.style1(target_image)
    s2 = m.style2(s1)
    c1 = m.content(s2)
    ResLayers(target_image, s1, s2, c1)
end

function add_style_loss(m::Model, style_image::AbstractArray{Float32,4})::Nothing
    c1 = Chain(Conv())
    return nothing
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
    style_patches₁ = patches_sampling(rtarget.style1)
    style_patches₂ = patches_sampling(rtarget.style2)

    current_patches₁ = patches_sampling(rcurrent.style1)
    current_patches₂ = patches_sampling(rcurrent.style2)

    conv₁ = Conv(style_patches₁, Flux.Zeros())
    conv₂ = Conv(style_patches₂, Flux.Zeros())

    max_response₁ = conv₁(current_patches₁)
    max_response₂ = conv₂(current_patches₂)

    snorm₁ = patches_norm(style_patches₁)
    snorm₂ = patches_norm(style_patches₂)
    cnorm₁ = patches_norm(current_patches₁)
    cnorm₂ = patches_norm(current_patches₂)

end
function content_loss(rcurrent::ResLayers, rtarget::ResLayers)::Float32
    sum((rcurrent.content - rtarget.content).^2)
end

function get_res_test()

    style = Images.load("img.jpg")
    content = Images.load("img1.jpg")
    sarr, carr = map(preprocess_array, img_to_arr.((style, content), ((250, 250),)))
    
    tarr = 2 .* rand(Float32, size(carr)) .- 1

    m = model()
    rstyle = m(sarr)
    # (cont_1, _, _) = m(carr)
    rtar = m(tarr)
    rtar, rstyle
end

function test()
    rtar, rstyle = get_res_test()
    l = style_loss(rtar, rstyle)
    rest = l.chain1(reshape(rtar.style1, size(rtar.style1)[begin:end - 1]))[1,:,:]
    println(size(rest))
    argm = argmax(rest, dims=1)
    println("Size argm: ", size(argm))
    cr = rest[1,1]
    tfil = rtar.style1[:,:,1]
    m1 = argm[1]
    sfil = rstyle.style1[:,:,m1[1]]
    vmul = sum(tfil .* sfil)
    nmul = sum(tfil * sfil)
    println("Cost on result: ", cr)
    println("vMult_sum (t.*s): ", vmul)
    println("Mult_sum (t*s): ", nmul)

    println(vmul in rest)

    hcat(tfil, sfil)
end        