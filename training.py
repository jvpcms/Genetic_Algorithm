import flapBird
import neuralNetwork

path = 'simulation_data/generation_'

# n° of entities at the end of generation, n° of generations, n° of new entities for each survivor, rate of mutation
number_survivors, children, generations, mutation_rate = 10, 100, 10, 1
population_size = number_survivors * children
layer_sizes = (4, 1)

population = neuralNetwork.generate_population(population_size, layer_sizes)

for g in range(1, generations + 1):

    print(f"Generation {g} ", end='')

    survivors, performance = flapBird.simulate(population, number_survivors)

    print("completed")
    print(f'Performance: {performance}\n')

    neuralNetwork.save_population(survivors, number_survivors, layer_sizes, path + f"{g}.json")
    population = neuralNetwork.new_generation(survivors, performance, mutation_rate, children, layer_sizes)
