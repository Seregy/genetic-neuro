import math
import random

import numpy as np
from tensorflow import keras

from com.seregy77.evnn.neural.network import Network
from com.seregy77.evnn.neural.utils import normalize_images, one_hot_encode
from com.seregy77.evnn.spea2.individual import Individual
from com.seregy77.evnn.spea2.network_parameter import LayerParameter


def init_dataset():
    # fashion_mnist = keras.datasets.fashion_mnist
    # # mnist = keras.datasets.mnist
    # (train_images, train_labels) = fashion_mnist.load_data()[0]
    # size = math.floor(len(train_images) * 0.8)
    # train_images = train_images[:size]
    # train_labels = train_labels[:size]
    #
    # train_images = normalize_images(train_images)
    # train_labels = one_hot_encode(train_labels)

    boston_housing = keras.datasets.boston_housing

    (train_images, train_labels), (test_images, test_labels) = boston_housing.load_data()

    mean = train_images.mean(axis=0)
    train_images -= mean
    std = train_images.std(axis=0)
    train_images /= std

    test_images -= mean
    test_images /= std

    return train_images, train_labels


def init_network():
    network = Network(trainable=False, output_classes=1)
    network.compile(loss='mean_squared_error', metrics=['mean_absolute_error'])
    return network


def more_or_equal(value1, value2):
    delta = 0.0001

    if value1 > value2 or abs(value1 - value2) < delta:
        return True

    return False


def compute_fitness(individuals, raw_fitness, density):
    overall_fitness = []
    for i in range(len(individuals)):
        fitness = raw_fitness[i] + density[i]
        individuals[i].fitness = fitness
        overall_fitness.append(fitness)

    return overall_fitness


def build_next_gen_archive(mixed_population, population_fitness):
    population_size = len(mixed_population)
    next_archive = []

    for i in range(population_size):
        if population_fitness[i] < 1:
            next_archive.append(mixed_population[i])

    return next_archive


def get_fitness(individual, population, population_fitness):
    for i in range(len(population)):
        if np.array_equal(individual, population[i]):
            return population_fitness[i]
    return 999


def build_mating_pool(population_size, archive, population, population_fitness):
    pool = []
    archive_size = len(archive)
    for i in range(population_size):
        first_index = random.randrange(archive_size)
        first_individual = archive[first_index]
        second_index = random.randrange(archive_size)
        second_individual = archive[second_index]
        if get_fitness(first_individual, population, population_fitness) < get_fitness(second_individual,
                                                                                       population,
                                                                                       population_fitness):
            pool.append(first_individual)
        else:
            pool.append(second_individual)
    return pool


def mix(current_population, archive):
    return current_population + archive


def get_non_dominated(individuals):
    result = []
    for i in range(len(individuals)):
        if individuals[i].fitness < 1:
            result.append(individuals[i])

    if not result:
        fittest_individual = sorted(individuals, key=lambda x: x.fitness)[0]
        result.append(fittest_individual)

    return result


def apply_crossover(parent1, parent2, probability):
    if random.uniform(0, 1) < probability:
        return parent1.cross(parent2)

    return parent1, parent2


class Spea2:
    _spea2_config = None
    _network_config = None
    _network = None
    _train_images = None
    _train_labels = None

    def __init__(self, spea2_config, network_config):
        self._spea2_config = spea2_config
        self._network_config = network_config
        (images, labels) = init_dataset()
        self._train_images = images
        self._train_labels = labels
        self._network = init_network()

    def init_weights(self):
        network = self._network_config
        layer_amount = len(network.layers)
        layer_params = []
        for i in range(layer_amount - 1):
            # Weights
            current_layer_size = network.layers[i]
            next_layer_size = network.layers[i + 1]

            weights = np.random.uniform(network.min_weight, network.max_weight, (current_layer_size, next_layer_size))
            biases = np.random.uniform(network.min_bias, network.max_bias, (next_layer_size,))

            layer_params.append(LayerParameter(weights, biases))

        return layer_params

    def init_population(self, size):
        result = []
        for i in range(size):
            individual = Individual(self.init_weights())
            result.append(individual)
        return result

    def objective1(self, individual):
        if individual.first_objective is None:
            self.init_objectives(individual)

        return individual.first_objective

    def init_objectives(self, individual):
        (loss, accuracy) = self.proceed_nn(individual.weights)
        individual.first_objective = -loss
        individual.second_objective = -accuracy

    def proceed_nn(self, weights):
        network = self._network
        network.assign_custom_weights(weights)
        train_loss, train_acc = network.evaluate(self._train_images, self._train_labels)
        return train_loss, train_acc

    def objective2(self, individual):
        if individual.second_objective is None:
            self.init_objectives(individual)

        return individual.second_objective

    def dominates(self, individual1, individual2):
        o11 = self.objective1(individual1)
        o12 = self.objective1(individual2)
        o21 = self.objective2(individual1)
        o22 = self.objective2(individual2)

        if more_or_equal(o11, o12) and more_or_equal(o21, o22) and (o11 > o12 or o21 > o22):
            return True

        return False

    def compute_s(self, mixed_population):
        s = []
        population_size = len(mixed_population)
        for i in range(population_size):
            current_individual = mixed_population[i]
            individual_strength = 0
            for j in range(population_size):
                j_individual = mixed_population[j]
                if self.dominates(current_individual, j_individual):
                    individual_strength = individual_strength + 1
            s.append(individual_strength)
        return s

    def compute_raw(self, population_strength, mixed_population):
        raw = []
        population_size = len(mixed_population)
        for i in range(population_size):
            current_individual = mixed_population[i]
            individual_raw = 0
            for j in range(population_size):
                j_individual = mixed_population[j]
                if current_individual == j_individual:
                    continue
                j_strength = population_strength[j]
                if self.dominates(j_individual, current_individual):
                    individual_raw = individual_raw + j_strength
            raw.append(individual_raw)
        return raw

    def compute_density(self, mixed_population):
        density = []
        population_size = len(mixed_population)
        k = math.sqrt(population_size)

        for i in range(population_size):
            current_individual = mixed_population[i]
            distances = []
            for j in range(population_size):
                individual_j = mixed_population[j]
                distance_j = self.compute_distance(current_individual, individual_j)
                distances.append(distance_j)
            distances.sort()

            sigma = distances[math.floor(k)]
            current_density = 1 / (sigma + 2)

            density.append(current_density)

        return density

    def compute_distance(self, individual1, individual2):
        o11 = self.objective1(individual1)
        o12 = self.objective1(individual2)
        o21 = self.objective2(individual1)
        o22 = self.objective2(individual2)

        dist1 = (o12 - o11) ** 2
        dist2 = (o22 - o21) ** 2

        return math.sqrt(dist1 + dist2)

    def adjust_next_gen_archive(self, next_archive, desired_size, mixed_population, population_fitness):
        current_size = len(next_archive)
        adjusted_archive = next_archive
        if desired_size == current_size:
            return adjusted_archive

        if desired_size > current_size:
            population_with_fitness = zip(mixed_population, population_fitness)
            sorted_population = sorted(population_with_fitness, key=lambda tup: tup[1])
            sorted_individuals, sorted_fitness = zip(*sorted_population)

            sorted_individuals_not_in_archive = [i for i in sorted_individuals if i not in next_archive]

            individuals_to_copy = desired_size - current_size
            for i in range(individuals_to_copy):
                adjusted_archive.append(sorted_individuals_not_in_archive[i])
            return adjusted_archive

        while len(adjusted_archive) > desired_size:
            current_size = len(adjusted_archive)
            minimal_distances = []
            for i in range(current_size):
                current_individual = next_archive[i]
                individual_distances = []
                for j in range(current_size):
                    j_individual = next_archive[j]
                    if i == j:
                        continue
                    distance = self.compute_distance(current_individual, j_individual)
                    individual_distances.append((j, distance))
                sorted_individuals_with_distances = sorted(individual_distances, key=lambda tup: tup[1])
                minimal_distances.append(sorted_individuals_with_distances[0])
            sorted_minimal_distances = sorted(minimal_distances, key=lambda tup: tup[1])
            index_to_remove = sorted_minimal_distances[0][0]
            adjusted_archive.pop(index_to_remove)
        return adjusted_archive

    def apply_gen_operators(self, mating_pool, crossover_probability, mutation_probability):
        mating_pool_size = len(mating_pool)
        crossed_population = []
        for i in range(0, mating_pool_size - 1, 2):
            new_individuals = apply_crossover(mating_pool[i], mating_pool[i + 1], crossover_probability)
            crossed_population.extend(new_individuals)

        if len(crossed_population) < mating_pool_size:
            crossed_population.append(mating_pool[mating_pool_size - 1])

        new_population = []
        for i in range(mating_pool_size):
            new_individual = self.apply_mutation(crossed_population[i], mutation_probability)
            new_population.append(new_individual)

        return new_population

    def apply_mutation(self, individual, probability):
        if random.uniform(0, 1) < probability:
            config = self._network_config
            return individual.mutate(config.min_weight, config.max_weight, config.min_bias, config.max_bias)

        return individual

    def execute(self):
        config = self._spea2_config
        population = self.init_population(config.population_size)
        archive = []

        print("Initial population:")
        for j in range(len(population)):
            print("Individual {}".format(j))
            print(population[j])

        for iteration in range(config.max_iterations):
            print("Iteration: {0}".format(iteration))
            rt = mix(population, archive)
            s = self.compute_s(rt)
            raw = self.compute_raw(s, rt)
            d = self.compute_density(rt)
            fitness = compute_fitness(rt, raw, d)
            archive_tmp = build_next_gen_archive(rt, fitness)
            adjusted_archive = self.adjust_next_gen_archive(archive_tmp, config.archive_size, rt, fitness)
            mating_pool = build_mating_pool(config.population_size, adjusted_archive, rt, fitness)
            population = self.apply_gen_operators(mating_pool, config.crossover_probability,
                                                  config.mutation_probability)
            archive = adjusted_archive

        return sorted(get_non_dominated(archive), key=lambda x: x.first_objective, reverse=True)
