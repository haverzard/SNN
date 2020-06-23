mutable struct Layer
    neurons_weights::Array{Float64,2}
    neurons_weights_total::Array{Float64,1}
    neurons_bias::Array{Float64,}
    neurons_count::Int
    inputs_count::Int
end

function create_layer(from::Int, to::Int)::Layer
    random_weights = rand(from, to)
    random_bias = rand(to)
    return Layer(random_weights, sum(random_weights, dims=1)[1:end], random_bias, to, from)
end

function goto_layer(input::Array{Float64,1}, layer::Layer)::Array{Float64,1}
    return sum(input .* layer.neurons_weights, dims=1)[1:end] .+ layer.neurons_bias
end