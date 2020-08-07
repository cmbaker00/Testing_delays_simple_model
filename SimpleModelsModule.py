import numpy as np
import matplotlib.pyplot as plt
from copy import copy

class InfectionDelay:
    def __init__(self, average_number_infections=2.53, pre_symp_infectious=2, symp_infectous=7.68,
                 pop_structure='uniform', pop_size=1000):
        self.average_number_infections = average_number_infections
        self.pre_symp_infectious = pre_symp_infectious
        self.symp_infectous = symp_infectous
        self.pop_structure = pop_structure
        self.pop_size = pop_size

        self.total_infectious = pre_symp_infectious + symp_infectous
        self.average_infectious_per_day = average_number_infections / self.total_infectious

        self.population = self.create_population()

    def create_population(self):
        if self.pop_structure == 'uniform':
            return np.zeros(self.pop_size) + self.average_number_infections
        if self.pop_structure == 'geometric':
            p = 1 / self.average_number_infections
            return np.random.geometric(p, self.pop_size) - 1
        if self.pop_structure == 'poisson':
            return np.random.poisson(self.average_number_infections, size = self.pop_size)
        raise ValueError('Population structure "{}" not implemented'.format(self.pop_structure))

    def pop_attack_vary_delay(self, min_lag=0, max_lag=None, max_infections=None, resolution=500):
        if max_lag is None:
            max_lag = self.total_infectious
        if max_infections is None:
            max_infections = np.inf

        lag_range = np.linspace(min_lag, max_lag, resolution)
        attack_rate_array = []
        for lag in lag_range:
            attack_rate_array.append(self.population_attack_rate(lag, max_infections))
        return attack_rate_array

    def population_attack_rate(self, lag=None, max_infections=np.inf):
        if lag is None:
            lag = self.total_infectious
        percent_reduction = lag/self.total_infectious
        new_popolation = self.population
        if max_infections < np.inf:
            new_popolation[new_popolation > max_infections] = max_infections
        new_popolation = new_popolation*percent_reduction

        r0 = self.calculate_r0(new_popolation)
        return self.calc_attack_rate(r0)

    @staticmethod
    def calculate_r0(input_population_infectiousness):
        return np.mean(input_population_infectiousness)

    @staticmethod
    def calc_attack_rate(r0):
        ave_r0 = np.mean(r0)
        if ave_r0 > 0:
            ar = 100 * (1 - 1 / ave_r0)
        elif ave_r0 == 0:
            ar = 0
        else:
            raise ValueError('Input r0 parameter is negative')
        if ar < 0:
            ar = 0
        return ar

if __name__ == "__main__":
    testing_dealy = InfectionDelay(pop_structure='uniform')
    print(np.mean(testing_dealy.population))
    print(testing_dealy.pop_attack_vary_delay(0))