using Flux
using Images
using CUDA
using Zygote
using Metalhead
using Statistics
using Plots
using IJulia

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

function gram_matrix(mat, normalize::Bool=false)
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
    weights = [i for i in 1:num]
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

function total_loss(model::vgg19, target_image, contentf, stylef; α=1e-3, β=1, γ=1)
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
    cat(r, g, b, dims=3)
end

function norm(x)
    means = [123.68, 116.779, 103.939 ]
    stds = [58.393, 57.12, 57.375]
    x .*= 255.0f0
    for i in 1:3
        x[:,:,i] .-= means[i]
        x[:,:,i] ./= stds[i]
    end
    x
end

function preprocess_array(img_arr)
    Flux.unsqueeze(imresize((2 * Float32.(img_arr) .- 1), (226, 226)), 4)
end
function deprocess_array(arr)
    nim = arr |> cpu
    nim .-= minimum(nim)
    nim ./= maximum(nim)
    if length(size(nim)) == 4
        return nim[:,:,:,1]
    end
    nim
end
function dep_array(arr)
    nim = arr |> cpu
    
    means = [123.68, 116.779, 103.939 ]
    stds = [58.393, 57.12, 57.375]
    for i in 1:3
        nim[:,:,i] .*= stds[i]
        nim[:,:,i] .+= means[i]
    end
    nim ./= 255.0f0
end

function arr_to_img(iar)
    arr = Array(iar)
    return RGB.([arr[:,:,i] for i in 1:3]...)
end

styles = Dict(
    :starry => "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/300px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
    :scream => "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg/1200px-Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg",
    :wave => "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Tsunami_by_hokusai_19th_century.jpg/800px-Tsunami_by_hokusai_19th_century.jpg",
    :giorgio => "https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/Claude_Monet%2C_Saint-Georges_majeur_au_cr%C3%A9puscule.jpg/1200px-Claude_Monet%2C_Saint-Georges_majeur_au_cr%C3%A9puscule.jpg",
    :naif => "https://instagram.fbcn11-1.fna.fbcdn.net/v/t51.2885-15/e35/s1080x1080/116798277_581494632757362_4406532804858518791_n.jpg?_nc_ht=instagram.fbcn11-1.fna.fbcdn.net&_nc_cat=106&_nc_ohc=7zx1nibyyZ8AX8Os3uD&tp=1&oh=b9301bf53db67ea59518f2d1ddec5024&oe=604ED2A8",
    :wheat => "https://d16kd6gzalkogb.cloudfront.net/magazine_images/Vincent-van-Gogh-Whaet-Field-with-Cypresses.-Image-via-wikimedia.org_.jpg",
    :sunrise => "https://www.artblr.com/upload/userfiles/images/Sunrise.jpg",
    :nature => "https://sothebys-com.brightspotcdn.com/dims4/default/cd1ca73/2147483647/strip/true/crop/859x859+71+0/resize/1200x1200!/quality/90/?url=http%3A%2F%2Fsothebys-brightspot.s3.amazonaws.com%2Fdotcom%2Fa4%2Fb2%2F1f464df248a8bbfe8466856b32bc%2Fcover-pics.jpg",
)

function get_images(style::Symbol=:starry)
    # cim = mktemp() do fn, f
    #     download("https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/Cumulus_Clouds_over_Yellow_Prairie2.jpg/1920px-Cumulus_Clouds_over_Yellow_Prairie2.jpg", fn)
    #     load(fn)
    # end
    cim = Images.load("img2.jpg")
    sim = mktemp() do fn, f
        download(styles[style], fn)
        load(fn)
    end

    return preprocess_array.(img_to_array.((cim, sim)))
end

function train(sty::Symbol=:starry, n::Int=50;plt::Bool=false,α=1e-3, β=1, γ=1, init::Symbol=:content,lr::Float32=0.02f0)
    GC.gc()
    CUDA.reclaim()
    @info "Loading model"
    vgg = @time vgg19() |> device
    @info "Loading images"
    content_image, style_image = @time get_images(sty) .|> device
    if init == :rand
        target_image = randn(size(content_image)) |> device
    elseif init == :content
        target_image = copy(content_image)
    elseif init == :style
        target_image = copy(style_image)
    elseif init == :diff
        target_image = content_image - style_image
    elseif init == :sum 
        target_image = (content_image + style_image) ./ 2
    end
    ipa = Flux.params(target_image)
    opt = ADAM(lr, (0.9, 0.999))

    stylef = style(vgg, style_image)
    contentf = content(vgg, content_image)

    loss(x1) = total_loss(vgg, x1, contentf, stylef; α=α, β=β, γ=γ)
    losses = Array{Float32,1}()
    for i = 1:n
        @info "Iteration $i"
        gs = gradient(ipa) do
            loss(target_image)
        end

        Flux.update!(opt, ipa, gs)
        ls = @show loss(target_image)
        push!(losses, ls)
        if plt
            IJulia.clear_output(true)
            display(arr_to_img(hcat(deprocess_array.((content_image, style_image, target_image))...)))
        end
    end
    cimage = hcat(deprocess_array.((content_image, style_image, target_image))...)
    bimage = hcat(dep_array.((content_image, style_image, target_image))...)
    cimage = clamp.(cimage, 0.0f0, 1.0f0)
    bimage = clamp.(bimage, 0.0f0, 1.0f0)
    Images.save("output.jpg", arr_to_img(cimage))
    Images.save("output2.jpg", arr_to_img(bimage))
    @info "Finished"
    if plt
        display(plot(losses))
        h, w, c, n = size(content_image)
        histogram(reshape(deprocess_array(content_image), h * w, c), α=0.3, layout=3, label="content")
        histogram!(reshape(deprocess_array(target_image), h * w, c), α=0.3, layout=3, label="target")
        display(histogram!(reshape(deprocess_array(style_image), h * w, c), α=0.3, layout=3, label="style"))
    end
    return target_image
end
