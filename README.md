# Genetic_Algorithm
Using a Genetic Algorithm to train Neural Networks in order to play a copy of the game Flappy Bird.

![demo_genetic_algorithm](https://github.com/user-attachments/assets/b33fc0f2-6eb8-4a90-bd17-c6c771c2f3ba)


## Neural Network
- Fixed topology of the Neural Networks, having 2 input nodes and one output node
- Input nodes are the horizontal distance to the bird and the next obstacle and the vertical distance between the bird and center of the gap
- Output node is normalized, varying from 0 to 1. When ever the output value reaches 0.5 the bird jumps. The normalization process will pe addressed further

### Parameters
- The weights of the network are initialized following a normal distribution
- The biases are 0 initially
