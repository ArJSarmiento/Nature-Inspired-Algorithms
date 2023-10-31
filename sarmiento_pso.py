"""
Programmer Information:
  Name: Arnel Jan E. Sarmiento
  Course: 3BSCS
  Student No.: 2021-05094

Program Description:
  Implement the Particle Swarm Optimization (PSO) algorithm using the
  Ackley function to find the fitness value of the generated population.

Fitness Function:
  f(x) = -a * exp(-b * sqrt(1/d * sum(x_i^2))) - exp(1/d * sum(cos(c * x_i))) + a + exp(1)
  Optimal Value: 0
  a = 20
  b = 0.2
  c = 2 * pi
  d = dimension of the solution
  lower_bound = -32.768
  upper_bound = 32.768
"""

import math
import random
from typing import Tuple


def ackley_function(solution: list) -> float:
    # Defining constants based on recommended values.
    a = 20
    b = 0.2
    c = 2 * math.pi

    # define the variables
    dimension = len(solution)
    sum1 = sum(i ** 2 for i in solution)
    sum2 = sum(math.cos(c * i) for i in solution)

    # compute the fitness value of the solution
    term1 = -a * math.exp(-b * math.sqrt(sum1 / dimension))
    term2 = math.exp(sum2 / dimension)
    return term1 - term2 + a + math.exp(1)


def generate_value(lower_bound: float, upper_bound: float) -> float:
    return lower_bound + (upper_bound - lower_bound) * random.random()


def update_velocity(vel_t0, best_so_far_sol, current_sol, best_sol_in_pop, dimension) -> list:
    # Equation: v_i(t+1) = w * v_i(t) + c1 * e1 * (p_i(t) - x_i(t)) + c2 * e2 * (p_g(t) - x_i(t))
    # w = constant from 0.5 to 0.9 (for this test we use 0.5)
    # c1 and c2 are alpha and beta respectively which is set to 2.0
    # e1 and e2 are random numbers between 0 and 1.0

    # term1 = w * v_i(t)
    term1 = [i * 0.50 for i in vel_t0]

    # term2 = c1 * e1 * (p_i(t) - x_i(t))
    # difference1 = p_i(t) - x_i(t)
    difference1 = [best_sol_in_pop[i] - current_sol[i] for i in range(dimension)]
    term2 = [i * 2.0 * random.random() for i in difference1]

    # term3 = c2 * e2 * (p_g(t) - x_i(t))
    # difference2 = p_g(t) - x_i(t)
    difference2 = [best_so_far_sol[i] - current_sol[i] for i in range(dimension)]
    term3 = [i * 2.0 * random.random() for i in difference2]

    return [a + b + c for a, b, c in zip(term1, term2, term3)]


def pso_algorithm(sol_t0: list, vel_t0: list) -> Tuple[list, list]:
    # define the variables
    num_of_population = 10
    population = []
    dimension = 5

    # the values of the bounds are based on the recommended value.
    lower_bound = -32.768
    upper_bound = 32.768

    # generate and print the initial population
    for i in range(num_of_population):
        solution = [generate_value(lower_bound, upper_bound) for _ in range(dimension)]
        population.append(solution)
        print(f'Solution {i + 1}: {solution}')

    # print the corresponding fitness value of each solution
    fitness_values = []
    print()
    for i in range(num_of_population):
        fitness_values.append(ackley_function(population[i]))
        print(f'Fitness value {i + 1}: {fitness_values[i]}')

    # find the index of the best solution
    # the best solution is the solution with the lowest fitness value
    best_solution_index = fitness_values.index(min(fitness_values))
    print(f'\nBest solution index: {best_solution_index + 1}')

    # we get the next position of every solution based on the best solution on the
    # population, the current position, and the best so far position of the particle/solution

    # update and print the velocity of the best solution
    vel_t1 = []
    print()
    for i in range(num_of_population):
        new_vel = update_velocity(vel_t0[i], sol_t0[i], population[i], population[best_solution_index], dimension)
        print(f'New Velocity {i + 1}: {new_vel}')
        vel_t1.append(new_vel)

    # update and print the position of every solution
    # Equation: x_i(t+1) = x_i(t) + v_i(t+1)
    sol_t1 = []
    print()
    for i in range(num_of_population):
        new_sol = [a + b for a, b in zip(population[i], vel_t1[i])]

        # check if the new solution is within the bounds
        for j in range(dimension):
            if new_sol[j] > upper_bound or new_sol[j] < lower_bound:
                new_sol[j] = population[i][j]

        # Mechanism to ensure that the new solution is better than the previous solution
        new_fitness_value = ackley_function(new_sol)
        if new_fitness_value < fitness_values[i]:
            sol_t1.append(new_sol)
        else:
            sol_t1.append(population[i])

        print(f'New Position {i + 1}: {sol_t1[i]}')

    # print the new fitness value of every solution
    new_fitness_values = []
    print()
    for i in range(num_of_population):
        new_fitness_value = ackley_function(sol_t1[i])
        new_fitness_values.append(new_fitness_value)
        print(f'New Fitness value {i + 1}: {new_fitness_value}')

    # determine if the fitness value improved or not
    print()
    for i in range(num_of_population):
        old_fitness_value = f'\tOld: {fitness_values[i]:0.6f}'
        new_fitness_value = f'\tNew: {new_fitness_values[i]:0.6f}'
        print(f'Fitness value {i + 1}:')
        print(old_fitness_value)
        print(new_fitness_value)
        print(f'\t{"Improved" if fitness_values[i] > new_fitness_values[i] else "Did not improve"}')

    return sol_t1, vel_t1


if __name__ == "__main__":
    test_iteration_count = 1

    sol_t0 = [[-5.556187890954266, -19.903305474229317, -7.466117953001646, 31.72649951339433, -19.514258213422313],
              [13.402350216320443, 8.841049853374408, -31.03351559948157, -5.571330455814866, 3.883961440719844],
              [-9.961709936059648, 22.298197422596708, -16.0098285552447, -16.122552092087673, 14.65229873233217],
              [17.32715104005449, 8.431794517332122, 7.126928831461463, 3.3930806365369293, -28.119546537915333],
              [-28.31933613855042, 20.627616969715824, -26.64044539180751, -17.97376885874183, -26.289865099341807],
              [24.944640392041173, 14.753364785366728, 23.416659235506216, 18.129559051841568, 27.699517663140547],
              [19.46220019167511, -16.902852037343852, 13.058740880897332, -16.82257116689396, 17.931792599203426],
              [19.599255730769535, 9.32524992979782, 9.578038422889712, -12.391846525661379, 1.2982656105083947],
              [-0.7664503306318622, -29.413157279363915, -30.40131419830396, 25.947020368046573,
               -23.219785119887813],
              [-23.92245707040432, -21.73899420904505, -27.69607776499083, -18.40294402742068, 16.62977494412241]]

    vel_t0 = [[2.92, 8.58, 9.9, 9.33, 6.3], [4.37, 7.5, 0.31, 6.52, 0.48], [9.37, 5.98, 6.93, 5.99, 5.35],
              [5.39, 0.56, 1.73, 3.8, 9.13], [3.47, 7.37, 0.42, 1.01, 6.56], [6.68, 7.61, 9.18, 5.24, 2.98],
              [5.63, 0.1, 1.22, 0.97, 6.54], [4.91, 5.49, 5.95, 0.7, 8.55], [0.29, 6.96, 8.81, 6.76, 5.35],
              [8.37, 2.84, 5.88, 1.95, 2.33]]

    for i in range(test_iteration_count):
        print(f'\nTest {i + 1}')
        sol_t1, vel_t1 = pso_algorithm(sol_t0, vel_t0)
        sol_t0 = sol_t1
        vel_t0 = vel_t1
