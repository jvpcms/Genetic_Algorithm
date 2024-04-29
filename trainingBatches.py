import falpBirdBatches
import neuralNetwork

path = 'simulation_data/generation_'

# n° of entities at the end of generation, n° of generations, n° of new entities for each survivor, rate of mutation
number_survivors, children, generations, mutation_rate = 10, 100, 10, 1
batch_size, n_bathces = 200, 5
population_size = number_survivors * children
layer_sizes = (4, 1)

population = neuralNetwork.generate_population(population_size, layer_sizes)

for g in range(1, generations + 1):

    fittest, best_performance = [], []

    for b in range(1, n_bathces + 1):
        survivors, performance = falpBirdBatches.simulate(population[(b-1)*batch_size:b*batch_size],
                                                   batch_size, number_survivors, g, b)
        if b == 1:
            fittest = survivors
            best_performance = performance
        else:
            new_fittest = [0]*number_survivors
            new_performance = [0]*number_survivors
            m1, m2 = number_survivors - 1, number_survivors - 1
            for i in range(number_survivors - 1, -1, -1):
                if performance[m1] > best_performance[m2]:
                    new_fittest[i] = survivors[m1]
                    new_performance[i] = performance[m1]
                    m1 -= 1
                else:
                    new_fittest[i] = fittest[m2]
                    new_performance[i] = best_performance[m2]
                    m2 -= 1
            fittest = new_fittest
            best_performance = new_performance

    neuralNetwork.save_population(fittest, number_survivors, layer_sizes, path + f"{g}.json")
    population = neuralNetwork.new_generation(fittest, best_performance, mutation_rate, children, layer_sizes)
