module SNN

include("Error.jl")

mutable struct Layer
    neurons_weights::Array{Float64,2}
    neurons_weights_total::Array{Float64,2}
    neurons_bias::Array{Float64,2}
    neurons_count::Int
    inputs_count::Int
end

function create_layer(from::Int, to::Int)::Layer
    random_weights = rand(from, to)
    random_bias = rand(1,to)
    return Layer(random_weights, sum(random_weights, dims=1) + random_bias, random_bias, to, from)
end

function activation(result::Array{Float64, 2}, maxs::Array{Float64,2})::Array{Float64, 2}
    result = result ./ maxs
    result[result .< 0] .= 0
    return result
end

function goto_layer(input::Array{Float64,2}, layer::Layer)::Array{Float64,2}
    result = (input * layer.neurons_weights) + layer.neurons_bias
    return activation(result, layer.neurons_weights_total)
end

mutable struct NeuralNetwork
    layers::Array{Layer, 1}
    total_layers::Int
    learning_rate::Float64
end

function create_neuralnetwork(mapping::Array{Int,1})::NeuralNetwork
    total_layers = length(mapping)-1
    if total_layers <= 0
        throw(GenerationError("Neural Network", "Bad mapping input"))
    end
    layers = Array{Layer,1}(undef, total_layers)
    for i in 1:total_layers
        layers[i] = create_layer(mapping[i], mapping[i+1])
    end
    return NeuralNetwork(layers, total_layers, 0.1)
end

function train_layer(layer::Layer, learning_rate::Int, errors::Array{Float64, 1})

end

function process(inputs::Array{Float64, 1}, neural::NeuralNetwork)

end

function train_neuralnetwork(inputs::Array{Float64,2}, neural::NeuralNetwork, expected_results::Array{Float64,2})
    layers_outputs = Array{Array{Float64,2},1}()
    
    layer_inputs = inputs
    for i in 1:neural.total_layers
        results = goto_layer(layer_inputs, neural.layers[i])
        push!(layers_outputs, results)
        layer_inputs = results
        print(layer_inputs)
    end
    print(layers_outputs)
end

end