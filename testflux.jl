using MLDatasets
using Flux: onehotbatch, Conv, relu, MaxPool, flatten, Dense, prod, logitcrossentropy, ADAM, onecold
using Base.Iterators: partition
using Flux
using Printf
using Statistics
using Torch: torch


const BATCHSIZE = 128
const W = 28
const H = 28

function build_model()
    Chain(
        Conv((3, 3), 1 => 16, pad = (1, 1), relu),
        MaxPool((2, 2)),
        Conv((3, 3), 16 => 32, pad = (1, 1), relu),
        MaxPool((2, 2)),
        Conv((3, 3), 32 => 32, pad = (1, 1), relu),
        MaxPool((2, 2)),
        flatten,
        Dense(prod((3, 3, 32, 1)), 10),
    )
end

function transform_x(x)
    w, h, n = size(x)
    return reshape(x, (w, h, 1, n))
end

data_loader(x, y) =
    Channel{Tuple{CuArray, Flux.OneHotArray}}() do c
        _, _, n = size(x)
        for p in partition(1:n, BATCHSIZE)
            batch_x = transform_x(x[:, :, p]) |> gpu
            batch_y = onehotbatch(y[p], 0:9) |> gpu
            put!(c, (gpu(batch_x), gpu(batch_y)))
        end
    end


function train_mnist()
    # train MNIST
    train_x, train_y = (MNIST.traintensor(Float32), MNIST.trainlabels())
    test_x, test_y = (MNIST.testtensor(Float32), MNIST.testlabels())

    # build model on GPU if possible
    model = build_model() |> gpu

    function accuracy(x, y, model)
        ŷ = onecold(cpu(model(gpu(transform_x(x))))) .- 1
        mean(ŷ .== y)
    end

    loss(x, y) = logitcrossentropy(model(x), y)
    opt = ADAM(3e-3)

    # ensure things are working
    assert_x = transform_x(train_x[:, :, 1:1]) |> gpu
    model(assert_x)

    best_acc = 0.0
    last_improvement = 0
    best_model = missing
    @info("Starting training from epoch 1")
    for epoch = 1:20
        # train on gpu
        train_loader = data_loader(train_x, train_y)
        Flux.train!(loss, Flux.params(model), train_loader, opt)
        # Calculate accuracy:
        acc = accuracy(test_x, test_y, model)

        @info(@sprintf("[%d]: Test accuracy: %.4f", epoch, acc))
        # If our accuracy is good enough, quit out.
        if acc >= 0.999
            @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
            return true
        end

        # If this is the best accuracy we've seen so far, save the model out
        if acc >= best_acc
            best_acc = acc
            last_improvement = epoch
            best_model = model
        end

    end
    return best_model
end
