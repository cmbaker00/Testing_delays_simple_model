import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from itertools import count
from TestingTargets import simple_exponential_growth


def calc_prob_detect_at_least_one(num_tests_per_day, prevalence_proportion, sensitivity):
    prob_zero_each_day = \
        [binom.cdf(0, tests, prob * sensitivity) for
         tests, prob in zip(num_tests_per_day, prevalence_proportion)]
    prob_zero_every_day = np.prod(prob_zero_each_day)
    prob_detect = 1 - prob_zero_every_day
    return prob_detect


def workplace_detection(num_people, growth_rate, number_of_times_testing_occurs_per_week,
                        proportion_workplace_tested_per_week, time_horizon=28, test_sensitivity=.85):
    num_infected = simple_exponential_growth(1, r_eff=growth_rate, num_days=time_horizon)
    proportion_infected = [n / num_people if n / num_people < 1 else 1 for n in num_infected]
    all_test_schedules = generate_all_test_schedules(num_people, number_of_times_testing_occurs_per_week,
                                                     proportion_workplace_tested_per_week, time_horizon)
    all_probs = [calc_prob_detect_at_least_one(test_schedule, proportion_infected, test_sensitivity) for
                 test_schedule in all_test_schedules]
    return all_probs, np.mean(all_probs), min(all_probs), max(all_probs)


def generate_all_test_schedules(num_people, number_of_times_testing_occurs_per_week,
                                proportion_workplace_tested_per_week, time_horizon):
    single_list = generate_single_test_schedule(num_people, number_of_times_testing_occurs_per_week,
                                                proportion_workplace_tested_per_week, time_horizon)
    return [shuffle_list_values(single_list, delay) for delay in range(7)]


def shuffle_list_values(input_list, amount_to_move):
    if amount_to_move > len(input_list):
        raise ValueError('amount_to_move must not be greater than the list length')
    if amount_to_move == 0:
        return input_list
    else:
        i = amount_to_move
        return input_list[-i:len(input_list)] + input_list[:len(input_list) - i]


def generate_single_test_schedule(num_people, number_of_times_testing_occurs_per_week,
                                  proportion_workplace_tested_per_week, time_horizon):
    test_days = None
    if number_of_times_testing_occurs_per_week == 1:
        test_days = (1, 0, 0, 0, 0, 0, 0)
    if number_of_times_testing_occurs_per_week == 2:
        test_days = (1, 0, 0, 0, 1, 0, 0)
    if number_of_times_testing_occurs_per_week == 3:
        test_days = (1, 0, 1, 0, 1, 0, 0)
    if number_of_times_testing_occurs_per_week == 4:
        test_days = (1, 1, 0, 1, 1, 0, 0)
    if number_of_times_testing_occurs_per_week == 5:
        test_days = (1, 1, 1, 1, 1, 0, 0)
    if number_of_times_testing_occurs_per_week is None:
        raise ValueError('Number of times testing per week must be an int from 1 to 5.')

    total_tests_per_week = np.round(num_people * proportion_workplace_tested_per_week)
    average_tests_per_testing_day = total_tests_per_week / number_of_times_testing_occurs_per_week
    average_tests_per_testing_day_lower = int(average_tests_per_testing_day)

    tests_per_day = np.array(test_days) * average_tests_per_testing_day_lower
    test_adds_up_flag = False
    test_day_index_current = 0
    while test_adds_up_flag is False:
        if sum(tests_per_day) == total_tests_per_week:
            test_adds_up_flag = True
        else:
            increment_testing_index = test_days.index(1, test_day_index_current)
            test_day_index_current += 1
            tests_per_day[increment_testing_index] += 1
    test_week_list = list(tests_per_day)
    num_weeks = int(time_horizon / 7)
    rem_days = int(np.round(time_horizon - num_weeks * 7))
    return test_week_list * num_weeks + test_week_list[0:rem_days]


if __name__ == "__main__":
    num_tests = [0, 0, 5, 5]
    prev = [1 / 100] * 4

    # print(calc_prob_detect_at_least_one(num_tests, prev,.85))
    simple_exponential_growth(1, r_eff=1.5, num_days=10)

    test_pr = workplace_detection(num_people=10, growth_rate=1.5,
                                  number_of_times_testing_occurs_per_week=5,
                                  proportion_workplace_tested_per_week=.5, time_horizon=7)

    print(test_pr[1])

    base_num_people = 50
    base_growth_rate = 1.5
    base_number_test_times = 3
    base_prop_per_week = .5
    base_time_horizon = 14
    base_test_sensitivity = .85

    workplace_size_plot = True
    workplace_testing_frequency_plot = True
    test_sensitivity_plot = True

    if test_sensitivity_plot:
        test_sensitivity_list = list(np.linspace(.75, .95, 10))
        pr_list = []
        for test_sens in test_sensitivity_list:
            pr = workplace_detection(base_num_people, growth_rate=base_growth_rate,
                                     number_of_times_testing_occurs_per_week=base_number_test_times,
                                     proportion_workplace_tested_per_week=base_prop_per_week,
                                     time_horizon=base_time_horizon,
                                     test_sensitivity=test_sens)
            pr_list.append(pr[1])
        plt.plot(test_sensitivity_list, pr_list)
        plt.xlabel('Test sensitivity')
        plt.ylabel('Probability of detection')
        plt.title('Test sensitivity')
        plt.savefig('Figures_workplace/workplace_test_sensitivity.png')
        plt.show()

    if workplace_testing_frequency_plot:
        test_freq_list = [1, 2, 3, 4, 5]
        pr_list = []
        for test_freq in test_freq_list:
            pr = workplace_detection(base_num_people, growth_rate=base_growth_rate,
                                     number_of_times_testing_occurs_per_week=test_freq,
                                     proportion_workplace_tested_per_week=base_prop_per_week,
                                     time_horizon=base_time_horizon,
                                     test_sensitivity=base_test_sensitivity)
            pr_list.append(pr[2])
        plt.plot(test_freq_list, pr_list)
        plt.xlabel('Number of times testing occurs per week')
        plt.ylabel('Minimum probability of detection')
        plt.title('Probability of detection, varying test frequency')
        plt.savefig('Figures_workplace/workplace_test_frequency.png')
        plt.show()

    if workplace_size_plot:
        pop_size = [2, 6, 10, 20, 30, 50, 100, 200]
        pr_list = []
        for pop in pop_size:
            pr = workplace_detection(pop, growth_rate=base_growth_rate,
                                     number_of_times_testing_occurs_per_week=base_number_test_times,
                                     proportion_workplace_tested_per_week=base_prop_per_week,
                                     time_horizon=base_time_horizon,
                                     test_sensitivity=base_test_sensitivity)
            pr_list.append(pr[1])
        plt.plot(pop_size, pr_list)
        plt.xlabel('Workplace size')
        plt.ylabel('Probability of detection')
        plt.title('Probability of detection with varying workplace size')
        plt.savefig('Figures_workplace/workplace_size.png')
        plt.show()
