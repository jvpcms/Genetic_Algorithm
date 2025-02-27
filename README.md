# Genetic_Algorithm

This repository details the methods used to train a Neural Network to play a version of the game Flappy Bird. In this project, I favored experimentation and discovery. The method described here is not set in stone nor is it intended to be the optimal approach for training. Rather, it is the result of iterative experimentation and personal learning. I am always looking to improve and deepen my understanding of these techniques.

![demo_genetic_algorithm](https://github.com/user-attachments/assets/b33fc0f2-6eb8-4a90-bd17-c6c771c2f3ba)

## Neural Network

### Topology
- A fixed topology with two input nodes and one output node.
- Input features:
  1. The horizontal distance between the bird and the next obstacle.
  2. The vertical distance between the bird and the center of the gap.
- Output node is a single value in the range [0, 1]. When the output value reaches 0.5 or higher, the bird performs a jump. The normalization procedure for this output is addressed further below.

### Parameters
- Weights of the network are initialized using a normal (Gaussian) distribution.
- Biases are initialized to 0.

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
# Selection snippet

```

### Crossover
The crossover operation is performed by selecting a random “node” (or parameter index) in the neural network’s parameter set and performing a linear partition on both parents' parameters. New offspring are created by combining the parameter segments from each parent.

```python
# Crossover snippet

```

### Mutation
During mutation, each parameter of a new individual’s network is considered for mutation based on a predefined mutation rate. When a parameter is selected to mutate, a small random value (sampled from a normal distribution) is added or subtracted from the existing parameter value.

```python
# Mutation snippet

```

## Tuning the Hyperparameters
During training, the maximum performance (fitness) of each generation is tracked. To tune the hyperparameters, I varied them individually and observed how the maximum performance evolves across generations.

### Stagnation
If the maximum performance value stops increasing for a certain number of consecutive generations, the training loop is halted. At this point, I adjust some of the hyperparameters (e.g., mutation rate, crossover method, population size) and begin a new training run to see if performance improves further.
