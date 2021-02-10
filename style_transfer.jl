using Flux
using Images
using CUDA
using Zygote
using Metalhead
using Statistics
using Plots

if has_cuda()
    CUDA.allowscalar(false)
    device = gpu
    @info "CUDA is on"
else
    device = cpu
    @info "CUDA is off"
end

mutable struct vgg19
    style1
    style2
    style3
    style4
    style5
    content1
end

Flux.@functor vgg19

function vgg19()
    vgg = VGG19().layers
    style1 = Chain(vgg[1:1]...)
    style2 = Chain(vgg[2:4]...)
    style3 = Chain(vgg[5:7]...)
    style4 = Chain(vgg[8:12]...)
    style5 = Chain(vgg[13:17]...)
    content1 = Chain(vgg[18:18]...)
    vgg19(style1, style2, style3, style4, style5, content1)
end

function content(m::vgg19, x)
    x |> m.style1 |> m.style2 |> m.style3 |> m.style4 |> m.style5 |> m.content1
end
function style(m::vgg19, x)
    s1 = m.style1(x)
    s2 = m.style2(s1)
    s3 = m.style3(s2)
    s4 = m.style4(s3)
    s5 = m.style5(s4)
    (s1, s2, s3, s4, s5)
end

function gram_matrix(mat, normalize=false)
    h, w, c, n = size(mat)
    nm = reshape(mat, h * w, c)
    gram = transpose(nm) * nm
    if normalize
        return gram ./ (2.0f0 * h * w * c)
    else
        return gram
    end
end

function content_loss(current, target)
    1 / 2 * sum((current - target).^2)
end

function style_loss(current_features, target_features)
    num = length(current_features)
    ls = 0f0
    weights = [1 / num for i in 1:num]
    for i in 1:num
        h, w, c, n = size(current_features[i])
        lloss = 1 / (4 * c^2 * (h * w)^2) * sum((gram_matrix(current_features[i]) .- gram_matrix(target_features[i]))^2)
        ls += weights[i] * lloss
    end
    return ls
end

function tv_loss(target_image)
    img = target_image
    h, w, c, q = size(img)
    ver_comp = sum((img[2:end,:,:,:] - img[1:end - 1,:,:,:]).^2)
    hor_comp = sum((img[:,2:end,:,:] - img[:,1:end - 1,:,:]).^2)
    1 / (4 * h * w * c) * (ver_comp + hor_comp).^2
end

function total_loss(model::vgg19, target_image, contentf, stylef)
    β = 1
    α = 1e-3
    γ = 1
    target_style = style(model, target_image)
    target_content = content(model, target_image)
    c_loss = content_loss(target_content, contentf)
    s_loss = style_loss(target_style, stylef) / 10
    t_loss = tv_loss(target_image)
    total = α * c_loss + β * s_loss + γ * t_loss
    println("%content: ", (α * c_loss) / total * 100)
    println("%style: ", (β * s_loss) / total * 100)
    println("%variation: ", (γ * t_loss) / total * 100)
    total
end

function img_to_array(img)
    r, g, b = red.(img), green.(img), blue.(img)
    im = cat(r, g, b, dims=3)
    re = imresize(im, (226, 226))
    return Flux.unsqueeze(Float32.(re), 4)
end

function arr_to_img(iar)
    arr = Array(iar)
    return RGB.([arr[:,:,i] for i in 1:3]...)
end

function get_images()
    cim = mktemp() do fn, f
        download("https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/Cumulus_Clouds_over_Yellow_Prairie2.jpg/1920px-Cumulus_Clouds_over_Yellow_Prairie2.jpg", fn)
        load(fn)
    end
    sim = mktemp() do fn, f
        download("https://ep01.epimg.net/cultura/imagenes/2020/04/05/babelia/1586104105_389523_1586104244_noticia_normal.jpg", fn)
        load(fn)
    end
    return img_to_array.((cim, sim))
end

function train(n::Int=25)
    GC.gc()
    CUDA.reclaim()
    @info "Loading model"
    vgg = @time vgg19() |> device
    @info "Loading images"
    content_image, style_image = @time get_images() .|> device
    target_image = copy(content_image)
    ipa = Flux.params(target_image)
    opt = ADAM(0.02, (0.9, 0.999))

    stylef = style(vgg, style_image)
    contentf = content(vgg, content_image)

    loss(x1) = total_loss(vgg, x1, contentf, stylef)
    losses = Array{Float32,1}()
    for i = 1:n
        @info "Iteration $i"
        gs = gradient(ipa) do
            loss(target_image)
        end

        Flux.update!(opt, ipa, gs)
        ls = @show loss(target_image)
        push!(losses, ls)
        display(arr_to_img(hcat(content_image, style_image, target_image ./ maximum(target_image))))
    end

    @info "Finished"
    display(plot(losses))
    return target_image
end
