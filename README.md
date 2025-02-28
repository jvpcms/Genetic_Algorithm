# Genetic_Algorithm

This repository details the methods used to train a Neural Network to play a version of the game Flappy Bird. In this project, I favored experimentation and discovery. The method described here is not set in stone nor is it intended to be the optimal approach for training. Rather, it is the result of iterative experimentation and personal learning. I am always looking to improve and deepen my understanding of these techniques.

![demo_genetic_algorithm](https://github.com/user-attachments/assets/b33fc0f2-6eb8-4a90-bd17-c6c771c2f3ba)

## Neural Network

### Topology
- A fixed topology with two input nodes and one output node.
- Input features:
  1. The horizontal distance between the bird and the next obstacle.
  2. The vertical distance between the bird and the center of the gap.
- Output node is a single value in the range [0, 1].

### Parameters
- Weights of the network are initialized using a normal (Gaussian) distribution.
- Biases are initialized to 0.

### Activaton Funciton
At each layer, the network computes the weighted sum of the outputs from the previous layer and adds a bias term to each node. This result passes through an activation function wich, for this implementation, is the sigmoid function:

```python
 def activation(x):
        return 1 / (1 + np.exp(-x))
```  

Because sigmoid outputs values in the [0, 1] range, it neatly fits the single output node that decides whether the bird should jump or not, based on a 0.5 threshold.

## Genetic Algorithm

- The initial population consists of 1000 randomly generated individuals.
- Each generation is simulated until 10 survivors remain.

### Fitness
The fitness function is defined by the total number of “loops” (or frames/intervals) the bird survives.

### Elitism
The top 10 survivors of each generation are copied directly into the next generation without undergoing either crossover or mutation. This ensures that high-performing solutions are not lost by random genetic operations.

### Selection
During reproduction, survivors with higher fitness scores in the previous generation have a higher probability of being selected as parents. One way to achieve this is to create a pool of individuals in which high performers appear more frequently and then randomly select from this pool. However, this can be memory-intensive. 

Instead, we can normalize the performance scores and then pick a random number between 0 and 1. If the normalized fitness of an individual is greater than or equal to this random number, that individual is selected for reproduction.

```python
def weighted_choice(weights):

    r = np.random.rand()
    for i in range(len(weights)):
        r -= weights[i]
        if r <= 0:
            return i
```

### Crossover
The crossover operation is performed by selecting a random “node” (or parameter index) in the neural network’s parameter set and performing a linear partition on both parents' parameters. New offspring are created by combining the parameter segments from each parent.

```python
def crossover(parent1, parent2, layer_sizes):
    child = parent2
    bias_split = int(np.random.rand() * sum(layer_sizes[:-1]))
    n_weights = 0

    for i in range(len(layer_sizes) - 1):
        n_weights += layer_sizes[i] * layer_sizes[i + 1]

    weight_split = int(np.random.rand() * n_weights)

    for bias in range(len(parent1.biases)):
        for b in range(len(parent1.biases[bias])):
            if bias_split > 0:
                child.biases[bias][b] = parent1.biases[bias][b]
            bias_split -= 1

    for weight in range(len(parent1.weights)):
        for line in range(len(parent1.weights[weight])):
            for col in range(len(parent1.weights[weight][line])):
                if weight_split > 0:
                    child.weights[weight][line][col] = parent1.weights[weight][line][col]
                weight_split -= 1

    return child
```

### Mutation
During mutation, each parameter of a new individual’s network is considered for mutation based on a predefined mutation rate. When a parameter is selected to mutate, a small random value (sampled from a normal distribution) is added or subtracted from the existing parameter value.

```python
def mutate(network, mutation_rate, layer_sizes):

    if np.random.rand() <= mutation_rate:
        weight_shapes = [(a, b) for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]
        weights = [np.random.standard_normal(s) / 100 for s in weight_shapes]
        biases = [np.random.standard_normal((s, 1)) / 100 for s in layer_sizes[1:]]

        weights = np.add(weights, network.weights)
        biases = np.add(biases, network.biases)

        return NeuralNetwork(layer_sizes, weights, biases)

    return network
```

## Tuning the Hyperparameters
During training, the maximum performance (fitness) of each generation is tracked. To tune the hyperparameters, I varied them individually and observed how the maximum performance evolves across generations.

### Stagnation
If the maximum performance value stops increasing for a certain number of consecutive generations, the training loop is halted. At this point, I adjust some of the hyperparameters (e.g., mutation rate, crossover method, population size) and begin a new training run to see if performance improves further.
