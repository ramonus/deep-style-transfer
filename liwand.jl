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

function patches_sampling(image::AbstractArray{Float32,4}, patch_size::Integer=3, stride::Integer=1)::AbstractArray{Float32,4}
    h, w = size(image)[1:2]
    patches = []
    for i in 1:stride:h - patch_size + 1 , j in 1:stride:w - patch_size + 1
        push!(patches, image[i:i + patch_size - 1,j:j + patch_size - 1,:,:])
    end
    cat(patches..., dims=4)
end

function patches_norm(patches::AbstractArray{Float32,4})::AbstractArray{Float32,1}
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
end

struct ResLayers
    input::AbstractArray{Float32,4}
    style1::AbstractArray{Float32,4}
    style2::AbstractArray{Float32,4}
    content::AbstractArray{Float32,4}
end

Flux.@functor Model

function model()::Model
    vgg = VGG19().layers
    style1 = Chain(vgg[1:7]...) |> device
    style2 = Chain(vgg[8:12]...) |> device
    content = Chain(vgg[13:13]...) |> device
    Model(
        style1,
        style2,
        content,
        )
end

function (m::Model)(target_image::AbstractArray{Float32,4})
    s1 = m.style1(target_image)
    # Losses on s1
    # p1 = patches_sampling(s1)
    # println(p1[:,:,1,100], m.style1_p[:,:,1,100])
    # n1 = patches_norm(p1)
    # l1 = conv(p1, m.style1_p, flipped=true)
    # l1 = reshape(l1, size(l1)[3:4])
    # l1 ./= [i * j for i in m.style1_n, j in n1]
    # m1 = argmax(l1, dims=1)
    # sloss1 = 0f0
    # for i in m1
    #     sloss1 += sum((p1[:,:,:,i[2]] - m.style1_p[:,:,:,i[1]]).^2)
    # end

    s2 = m.style2(s1)
    # Losses in s2
    # p2 = patches_sampling(s2)
    # n2 = patches_norm(p2)
    # l2 = conv(p2, m.style2_p, flipped=true)
    # l2 = reshape(l2, size(l2)[3:4])
    # l2 ./= [i * j for i in m.style2_n, j in n2]
    # m2 = argmax(l2, dims=1)
    # sloss2 = 0f0
    # for i in m2
    #     sloss2 += sum((p2[:,:,:,i[2]] - m.style2_p[:,:,:,i[1]]).^2)
    # end
    c1 = m.content(s2)
    ResLayers(
        target_image,
        s1,
        s2,
        c1,
        )
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

function content_loss(current::AbstractArray{Float32,4}, target::AbstractArray{Float32,4})::Float32
    sum((current - target).^2)
end

function style_loss(current_patches::AbstractArray{Float32,4}, current_norms::AbstractArray{Float32,1}, target_patches::AbstractArray{Float32,4}, target_norms::AbstractArray{Float32,1})::Float32
    l1 = Conv(reverse(reverse(target_patches, dims=1), dims=2), Flux.Zeros())(current_patches)
    l1 = reshape(l1, size(l1)[3:4])
    l1 ./= [i * j for i in target_norms, j in current_norms]
    m = argmax(l1, dims=1)
    loss = 0f0
    for i in m
        loss += sum((current_patches[:,:,:,i[2]] - target_patches[:,:,:,i[1]]).^2)
    end
    loss
end

function total_loss(content::Tuple{T,T}, styles::Tuple{Tuple{T,T},Tuple{T,T}}, norms::Tuple{Tuple{T2,T2},Tuple{T2,T2}})::Float32 where {T <: AbstractArray{Float32,4},T2 <: AbstractArray{Float32,1}}
    target_content, content = content
    ((ptstyle1, ptstyle2), (pstyle1, pstyle2)) = styles
    ((ntstyle1, ntstyle2), (nstyle1, nstyle2)) = norms
    ls1 = style_loss(pstyle1, nstyle1, ptstyle1, ntstyle1)
    ls2 = style_loss(pstyle2, nstyle2, ptstyle2, ntstyle2)
    cl = content_loss(content, target_content)
    return cl + ls1 + ls2
end

function get_test()

    style = Images.load("img.jpg")
    content = Images.load("img1.jpg")
    sarr, carr = map(preprocess_array, img_to_arr.((style, content), ((250, 250),)))
    
    tarr = 2 .* rand(Float32, size(carr)) .- 1

    m = model()

    m(sarr)
end

function train(epochs::Integer=1)
    imc = Images.load("david.jpg")
    ims = Images.load("temp.jpg")
    carr, sarr = map(preprocess_array, img_to_arr.((imc, ims), (250, 250))) .|> device
    println("Images loaded")
    m = model()
    rc = m(carr)
    rs = m(sarr)
    println("Model executed")
    s1patches = patches_sampling(rs.style1)
    s1norm = patches_norm(s1patches)
    s2patches = patches_sampling(rs.style2)
    s2norm = patches_norm(s2patches)
    println("Patches made")
    tim = 2 .* rand(Float32, 250, 250, 3, 1) .- 1 |> device
    pa = Flux.params(tim)
    opt = ADAM(0.02)
    for i = 1:epochs
        println("Iteration: $i")
        gs = gradient(pa) do
            r = m(tim)
            println("Model executed")
            s1p = patches_sampling(r.style1)
            s1n = patches_norm(s1p)
            s2p = patches_sampling(r.style2)
            s2n = patches_norm(s2p)
            println("Patched")
            content = (r.content, rc.content)
            styles = (
                (s1p, s1patches),
                (s2p, s2patches)
            )
            norms = (
                (s1n, s1norm),
                (s2n, s2norm)
            )
            total_loss(content, styles, norms)
        end
        Flux.update!(opt, pa, gs)
    end
    tim
end        