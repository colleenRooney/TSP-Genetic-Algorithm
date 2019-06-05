import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

class TSP:
    def __init__(self, distance_table, pop_size, optimal=None):
        self.distance_table = distance_table
        self.ncities = self.distance_table.shape[0]
        self.pop_size = pop_size
        self.pop = self.random_first_gen(pop_size)
        self.pmf = self. generate_pmf() 

    def generate_pmf(self):
        pmf = []
        for route in self.pop:
            pmf.append(1/route.distance)
        pmf = pmf/np.sum(pmf)
        return pmf

    def random_first_gen(self, pop_size):
        pop = []
        for i in range(0, pop_size):
            total = 0
            rand_route = np.random.permutation(self.ncities)
            dist = self.calculate_distance(rand_route)
            if i == 0:
                self.min_dist = dist
            else:
                if dist < self.min_dist: self.min_dist = dist
            pop.append(Route(rand_route, dist))
        return pop

    def calculate_distance(self, route):
        if route is None:
            return None
        dist = 0
        for i, a in enumerate(route):
            j = 0 if i == (self.ncities-1) else (i+1)
            b = route[j]
            d = self.distance_table[a, b]
            dist += d
        return dist

    def generate_mating_pool(self, size, elite):
        n = int(elite * size)
        pop_a = np.random.choice(self.pop, size-n, p=self.pmf)
        pop_b = sorted(self.pop, key=lambda x: x.distance, reverse=False)
        pop_b = np.array(pop_b[:n])
        return np.append(pop_a, pop_b)

    def next_gen(self, size, elite, mutation_rate):
        pop = []
        mating_pool = self.generate_mating_pool(size, elite)
        for i in range(0, self.pop_size):
            parents = np.random.choice(mating_pool, 2)
            route = self.crossover(parents[0], parents[1])
            dist = self.calculate_distance(route)
            if i == 0:
                self.min_dist = dist
            else:
                if dist < self.min_dist: self.min_dist = dist
            pop.append(Route(route, dist))
        for route in pop:
            self.mutate(route.route, mutation_rate)
        self.pop = pop
        self.generate_pmf()

    def mutate(self, route, mutation_rate):
        for a in range(0, self.ncities):
            if(random.random() < mutation_rate):
                b = random.randint(0, self.ncities-1)
                route[a], route[b] = route[b], route[a]

    def crossover(self, X, Y):
        points = [random.randint(0, self.ncities-1) for i in range(2)]
        points.sort()
        baby_route = [None] * (self.ncities)
        points[1] = 48
        babyX = X.route[points[0]:points[1]]
        for i in range(points[0], points[1]):
            baby_route[i] = X.route[i]
        i = 0
        for city in Y.route:
            if i == points[0]:
                i = points[1]
            if city not in babyX:
                baby_route[i] = city
                i += 1

        baby_route = np.array(baby_route)
        return baby_route


class Route:
    def __init__(self, city_list, distance):
        self.route = city_list
        self.distance = distance



# load distance table
dist = pd.read_csv('att48_d.txt', delim_whitespace=True, header=None).values

# load optimal solution, convert to 0 based indexing
optimal = pd.read_csv('att48_s.txt', header=None).values
optimal = optimal - 1

gen = TSP(dist, 100, optimal)
i = 0
for i in range(0, 1000):
    gen.next_gen(25, .4, 0.01)
    print(gen.min_dist)


#fig = plt.figure()
#plt.plot(rand_dist, label="random walks")
#plt.axhline(y=opt, color='r', linestyle='-')
#plt.show()
