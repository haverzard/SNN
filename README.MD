# Simple Neural Network
A simple neural network made from scratch in Julia. The project is just for fun and learning, so any feedback will be appreciated.

## Usage
### Import
```include("src/main.py")```

### Create neural network
```n = SNN.create_neuralnetwork(layer_mapper)```

layer_mapper is an array consisted of number, for example [5,4,1], which means that the neural network will receive 5 neurons input and it will map the input into 1 neuron output through a hidden layer which have 4 neurons.

### Train your neural network
```SNN.train_neuralnetwork(input, neural_network, expected)```

For the example neural network above, you can use input matrix with size n x 5 and expected matrix with size n x 1. It has the same idea of batch.

### Use your neural network
```SNN.pass_data(input, n)```

## References
- Neural network: http://neuralnetworksanddeeplearning.com