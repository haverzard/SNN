module SNN

include("Error.jl")

mutable struct Layer
    neurons_weights::Array{Float64, 2}
    neurons_bias::Array{Float64, 2}
    neurons_count::Int
    inputs_count::Int
end

function create_layer(from::Int, to::Int)::Layer
    random_weights = rand(to, from)
    random_bias = rand(to, 1)
    return Layer(random_weights, random_bias, to, from)
end

# We use the sigmoid activation function for now
function activation(result::Array{Float64, 2})::Array{Float64, 2}
    result = 1 ./ (1 .+ exp.(-result))
    return result
end

function goto_layer(input::Array{Float64, 2}, layer::Layer)::Array{Float64, 2}
    result = (layer.neurons_weights * input) .+ layer.neurons_bias
    return activation(result)
end

mutable struct NeuralNetwork
    layers::Array{Layer, 1}
    total_layers::Int
    learning_rate::Float64
end

function create_neuralnetwork(mapping::Array{Int, 1})::NeuralNetwork
    total_layers = length(mapping)-1
    if total_layers <= 0
        throw(GenerationError("Neural Network", "Bad mapping input"))
    end
    layers = Array{Layer, 1}(undef, total_layers)
    for i in 1:total_layers
        layers[i] = create_layer(mapping[i], mapping[i+1])
    end
    return NeuralNetwork(layers, total_layers, 0.8)
end

function backprop1(neurons::Array{Float64, 2}, expected::Array{Float64, 2})::Array{Float64, 2}
    delta_activation = neurons .* (1 .- neurons)
    delta_c = neurons .- expected
    return delta_c .* delta_activation
end

function backprop2(neurons::Array{Float64, 2}, layer_weigths::Array{Float64, 2}, errors::Array{Float64, 2})::Array{Float64, 2}
    delta_activation = neurons .* (1 .- neurons)
    return (transpose(layer_weigths) * errors) .* delta_activation
end

function backprop4(neurons::Array{Float64,2}, errors::Array{Float64, 2})::Array{Float64, 2}
    return errors * transpose(neurons)
end

function pass_data(input::Array{Float64, 2}, neural::NeuralNetwork)::Array{Float64, 2}
    layer_input = input
    for i in 1:neural.total_layers
        results = goto_layer(layer_input, neural.layers[i])
        layer_input = results
    end
    return layer_input
end

function train_neuralnetwork(inputs::Array{Float64, 2}, neural::NeuralNetwork, expected_results::Array{Float64, 2})
    layers_outputs = Array{Array{Float64, 2}, 1}()
    layers_errors = Array{Array{Float64, 2}, 1}(undef, neural.total_layers)
    (n, _) = size(inputs)
    L = neural.total_layers
    
    layer_inputs = inputs
    push!(layers_outputs, inputs)
    for i in 1:L
        results = goto_layer(layer_inputs, neural.layers[i])
        push!(layers_outputs, results)
        layer_inputs = results
    end

    # Getting the last layer error
    layers_errors[L] = backprop1(layers_outputs[L+1], expected_results) ./ n

    # Getting the other layer error
    for i in reverse(1:L-1)
        layers_errors[i] = backprop2(layers_outputs[i+1], neural.layers[i+1].neurons_weights, layers_errors[i+1]) ./ n
    end

    # Calculate gradient descent and learn
    for i in 1:L
        neural.layers[i].neurons_bias = neural.layers[i].neurons_bias .- (sum(layers_errors[i], dims=2) .* neural.learning_rate)
        neural.layers[i].neurons_weights = neural.layers[i].neurons_weights .- (backprop4(layers_outputs[i], layers_errors[i]) .* neural.learning_rate)
    end
end

end
