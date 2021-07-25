using MLDatasets
using Flux: onehotbatch, Conv, relu, MaxPool, flatten, Dense, prod, logitcrossentropy, ADAM, onecold
using Base.Iterators: partition
using Flux
using Printf
using Statistics

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

data_loader(x, y) =
    Channel() do c
        w, h, n = size(x)
        for p in partition(1:n, BATCHSIZE)
            b = size(p)[1]
            batch_x = x[:, :, p]
            batch_x = reshape(batch_x, (w, h, 1, b))
            batch_y = onehotbatch(y[p], 0:9)
            put!(c, (batch_x, batch_y))
        end
    end


function train_mnist()
    # train MNIST
    train_x, train_y = (MNIST.traintensor(Float32), MNIST.trainlabels())
    test_x, test_y = (MNIST.testtensor(Float32), MNIST.testlabels())
    train_loader = data_loader(train_x, train_y)
    model = build_model()

    function accuracy(x, y, model)
        ŷ = onecold(model(reshape(x, (W, H, 1, :)))) .- 1
        mean(ŷ .== y)
    end

    loss(x, y) = logitcrossentropy(model(x), y)
    opt = ADAM(3e-3)

    # ensure things are working
    model(first(train_loader)[1])

    best_acc = 0.0
    last_improvement = 0
    best_model = missing
    for epoch = 1:20
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
