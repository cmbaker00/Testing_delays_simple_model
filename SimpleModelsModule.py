import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from Simple_epi_models.ODE_Models import SIR_model_R0
import pandas as pd
from functools import lru_cache
from scipy.interpolate import interp2d


class InfectionDelay:
    def __init__(self, average_number_infections=2.53,
                 pre_symp_infectious=2, symp_infectous=7.68,
                 pop_structure='uniform', pop_size=1000):
        self.average_number_infections = average_number_infections
        self.pre_symp_infectious = pre_symp_infectious
        self.symp_infectous = symp_infectous
        self.pop_structure = pop_structure
        self.pop_size = pop_size

        self.total_infectious = pre_symp_infectious + symp_infectous
        self.average_infectious_per_day = average_number_infections \
                                          / self.total_infectious

        self.population = self.create_population()

    def create_population(self):
        if self.pop_structure == 'uniform':
            return np.zeros(self.pop_size) + self.average_number_infections
        if self.pop_structure == 'geometric':
            p = 1 / self.average_number_infections
            return np.random.geometric(p, self.pop_size) - 1
        if self.pop_structure == 'poisson':
            return np.random.poisson(self.average_number_infections,
                                     size = self.pop_size)
        raise ValueError('Population structure "{}" not implemented'.
                         format(self.pop_structure))

    def pop_attack_vary_delay(self, min_lag=0, max_lag=None,
                              max_infections=None, resolution=500):
        if max_lag is None:
            max_lag = self.total_infectious
        if max_infections is None:
            max_infections = np.inf

        lag_range = np.linspace(min_lag, max_lag, resolution)
        attack_rate_array = []
        for lag in lag_range:
            attack_rate_array.append(
                self.population_attack_rate(lag, max_infections)
            )
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
            ar = SIR_model_R0(ave_r0).est_total_infected()
        elif ave_r0 == 0:
            ar = 0
        else:
            raise ValueError('Input r0 parameter is negative')
        if ar < 0:
            ar = 0
        return ar


class TestOptimisation:
    def __init__(self, population=(1000, 10000, 100000),
                 pre_test_probability=(.4, .04, .004),
                 onward_transmission=(3, 4, 2, .5),
                 routine_capacity=5000,
                 priority_capacity_proportion=.1,
                 priority_queue=True,
                 routine_tat=1,
                 tat_at_fifty_percent_surge=2
                 ):
        self.population = population
        self.close_contact, self.symptomatic, self.asymptomatic = population
        self.onward_transmission = onward_transmission
        self.pre_test_by_indication = pre_test_probability

        self.routine_capacity = routine_capacity
        self.priority_capacity = routine_capacity*priority_capacity_proportion
        self.priority_queue = priority_queue

        self.routine_tat = routine_tat
        self.tat_surge = tat_at_fifty_percent_surge

        if not 0 < priority_capacity_proportion < 1:
            raise ValueError(f'Priority capacity proportion must be between '
                             f'0 and 1. The input value was '
                             f'{priority_capacity_proportion}.')

    def turn_around_time(self, tests, priority_queue=False):
        return self.function_turn_around_time(priority_queue)(tests)

    @lru_cache()
    def function_turn_around_time(self, priority_queue=False):
        routine_capacity = self.routine_capacity
        tat = self.routine_tat
        if priority_queue:
            tat_surge = tat
        else:
            tat_surge = self.tat_surge
        return lambda x: tat if x < routine_capacity else tat + (tat_surge-tat)*((x-routine_capacity)**2)/((routine_capacity*.5)**2)

    @lru_cache()
    def load_test_delay_data(self):
        data = pd.read_csv('testing_delay_kretzhcmar_table_2.csv')
        x = np.arange(0,6)
        y = np.arange(0,8)
        z = np.zeros([len(y), len(x)])
        z[:, :4] = np.array(data)[:, 1:5]
        z[:, -1] = np.array(data)[:, -1]
        z[:, -2] = (z[:, -1] + z[:, -3])/2 # assume that contact tracing with greater than 5 day delay is useless
        return interp2d(x, y, z)

    def test_delay_effect_on_percent_future_infections(self, swab_delay=0., result_delay=2.):
        return self.load_test_delay_data()(result_delay, swab_delay)

    @lru_cache()
    def create_pre_test_proabability_array(self):
        array = np.zeros([4,3])
        for i in range(3):
            array[:,i] = self.pre_test_by_indication[i]
        return array

    @lru_cache()
    def create_onward_transmission_array(self):
        array = np.zeros([4, 3])
        for i in range(3):
            array[:, i] = self.onward_transmission
        return array

    @lru_cache()
    def create_population_groups(self):
        array = np.zeros([4,3])
        relative_transmission = np.array([i/sum(self.onward_transmission) for
                                 i in self.onward_transmission])
        for i in range(3):
            array[:,i] = self.population[i]*relative_transmission

        return array

    def create_expected_onward_transmission_array(self):
        onward = self.create_onward_transmission_array()
        prob = self.create_pre_test_proabability_array()
        return onward*prob

    def create_transmission_tested_array(self, result_delay=2,
                                                  swab_delay=1):
        onward = self.create_onward_transmission_array()
        transmission_reduction = self.test_delay_effect_on_percent_future_infections(
            result_delay=result_delay,
            swab_delay=swab_delay)
        return onward*(1-transmission_reduction)

    def create_expected_transmission_tested_array(self, result_delay=2,
                                                  swab_delay=1):
        onward = self.create_transmission_tested_array(
            result_delay=result_delay,
            swab_delay=swab_delay)

        prob = self.create_pre_test_proabability_array()
        return onward*prob

    def benefit_of_test(self, result_delay, swab_delay = 1):
        # expected_trans = self.create_expected_onward_transmission_array()
        expected_trans = self.create_expected_transmission_tested_array(
            result_delay=np.inf, swab_delay=swab_delay
        )
        expected_tested_trans = self.create_expected_transmission_tested_array(
            result_delay=result_delay, swab_delay=swab_delay
        )
        return expected_trans - expected_tested_trans
        # expected_onward_infection = self.

    def plot_benefit_as_function_delay(self, swab_delay=1):
        result_delay = np.linspace(0,5,1000)
        benefit_array = []
        for res_del in result_delay:
            benefit_array.append(self.benefit_of_test(
                result_delay=res_del,
                swab_delay=swab_delay))
        benefit_array = np.array(benefit_array)
        for i in range(3):
            plt.plot(result_delay, benefit_array[:,:,i])
        plt.show()

    def allocate_tests(self, num_tests=1000,
                       result_delay=1,
                       swab_delay=1):
        benefit_array = self.benefit_of_test(result_delay=result_delay,
                             swab_delay=swab_delay)
        tests_remaining = num_tests
        num_tests_by_group = np.zeros(np.shape(benefit_array))
        pop_per_group = self.create_population_groups()
        while tests_remaining > 0:
            max_benefit = np.max(benefit_array)
            max_benefit_location = benefit_array == max_benefit
            total_pop_in_best_groups = np.sum(pop_per_group[max_benefit_location])
            if total_pop_in_best_groups < tests_remaining:
                num_tests_by_group[max_benefit_location] = pop_per_group[max_benefit_location]
            else:
                num_tests_by_group[max_benefit_location] = pop_per_group[max_benefit_location]*\
                                                           tests_remaining/total_pop_in_best_groups
            tests_remaining -= total_pop_in_best_groups
            benefit_array[max_benefit_location] = -1
        return num_tests_by_group

    def estimate_total_tranmission(self, test_allocation,
                                   result_delay=1, swab_delay=1):


        pop_per_group = self.create_population_groups()
        pop_untested = pop_per_group - test_allocation

        priority_tat = self.turn_around_time(tests=1)
        if self.priority_queue:
            total_tests = np.sum(test_allocation)
            num_priority_tests = min([total_tests, self.priority_capacity])
            test_allocation_priority = self.allocate_tests(num_tests=int(num_priority_tests),
                                                           result_delay=priority_tat,
                                                           swab_delay=swab_delay)
        else:
            test_allocation_priority = np.zeros(test_allocation.shape)

        test_allocation -= test_allocation_priority
        test_allocation[test_allocation<0] = 0

        exp_transmission_priority_test = self.create_expected_transmission_tested_array(
            result_delay=priority_tat,
            swab_delay=swab_delay
        )

        exp_transmission_test = self.create_expected_transmission_tested_array(
            result_delay=result_delay,
            swab_delay=swab_delay
        )
        exp_transmission_notest = self.create_expected_transmission_tested_array(
            result_delay=np.inf,
            swab_delay=swab_delay
        )

        untested_transmission = np.sum(pop_untested*exp_transmission_notest)
        tested_transmission = np.sum(test_allocation*exp_transmission_test)
        priority_tested_transmission = np.sum(test_allocation_priority*exp_transmission_priority_test)
        return tested_transmission + untested_transmission + priority_tested_transmission

    def estimate_transmission_with_testing(self, num_test, swab_delay=1, priority_queue=False):
        tat = self.turn_around_time(num_test) #todo: remove priority queue from turn_around_time?
        test_allocation = self.allocate_tests(num_tests=num_test,
                            result_delay=tat, swab_delay=swab_delay)
        return self.estimate_total_tranmission(test_allocation,
                                        result_delay=tat, swab_delay=swab_delay)

    @lru_cache()
    def generate_onward_transmission_with_tests(self, swab_delay=1):
        num_test_array = range(self.routine_capacity*2)
        transmission = []
        for num_tests in num_test_array:
            transmission.append(self.estimate_transmission_with_testing(
                num_test=num_tests, swab_delay=swab_delay, priority_queue=self.priority_queue
            ))
        transmission = np.array(transmission)/\
                       np.sum(np.array(self.population) * np.array(self.pre_test_by_indication))
        return num_test_array, transmission


    def plot_transmission_with_testing(self, swab_delay=1):
        plt.plot(*self.generate_onward_transmission_with_tests(swab_delay=swab_delay))
        plt.xlabel('Number of tests')
        plt.ylabel('Average onward transmission')
        if self.priority_queue:
            plt.title(f'Test capacity = {self.routine_capacity}'
                      f', with priority testing')
            plt.savefig(f'Test_capacity_{self.routine_capacity}'
                        f'_priority_testing.png')
        else:
            plt.title(f'Test capacity = {self.routine_capacity}')
            plt.savefig(f'Test_capacity_{self.routine_capacity}.png')
        plt.show()

    def optimal_test_amount(self):
        num_test_array, transmission = self.generate_onward_transmission_with_tests()
        opt_test = num_test_array[np.where(transmission == min(transmission))[0][0]] #todo not sure what would happen if there were two values at the min
        # if len(opt_test) > 1:
        #     opt_test = opt_test[0]
        num_tests_by_group = self.allocate_tests(num_tests=opt_test)
        tests_by_indication = np.sum(num_tests_by_group, axis=0)
        return opt_test, tests_by_indication, num_tests_by_group

if __name__ == "__main__":
    # test_optim = TestOptimisation(priority_queue=True)
    # TestOptimisation(priority_queue=True).estimate_transmission_with_testing(8000, 1)
    # # test_optim.create_onward_transmission_array()
    # # test_optim.plot_benefit_as_function_delay(swab_delay=1)
    # # print(test_optim.benefit_of_test(0))
    # # allocation = test_optim.allocate_tests(10000)
    # # print(test_optim.estimate_transmission_with_testing(0))
    # test_optim.plot_transmission_with_testing()

    test_optim = TestOptimisation(priority_queue=True)
    # TestOptimisation(priority_queue=False).plot_transmission_with_testing()
    # test_optim.plot_transmission_with_testing()
    tests, test_array = test_optim.optimal_test_amount()

    # testing_delay = InfectionDelay(pop_structure='uniform')
    # print(np.mean(testing_delay.population))
    # print(testing_delay.pop_attack_vary_delay(min_lag=0))
    # plt.plot(testing_delay.pop_attack_vary_delay(min_lag=0))
    # plt.show()