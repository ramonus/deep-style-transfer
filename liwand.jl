using Flux
using Images
using CUDA
using Zygote
using Metalhead
using Statistics
using LinearAlgebra
using Plots

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
    (c1, s1, s2)
end

function img_to_arr(img::Array{<:AbstractRGB,2})
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

function style_loss(current_style::Tuple{arrOcarr{Float32,4}}, target_style::Tuple{arrOcarr{Float32,4}})::Float32
    loss = 0f0

    for i in 1:length(current_style)
    end
end
function content_loss(current_content::arrOcarr{Float32,4}, target_content::arrOcarr{Float32,4})::Float32
    sum((current_content - target_content).^2)
end