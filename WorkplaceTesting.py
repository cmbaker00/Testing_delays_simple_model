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

    testing_schedule_breakdown = [[divmod(tests, num_people) for tests in test_schedule] for test_schedule in all_test_schedules]
    test_schedule_full_workplace, remaining_tests_schedule = ([[daily_tests[index] for daily_tests in current_test_schedule] for current_test_schedule in testing_schedule_breakdown] for index in range(2))
    if np.any(np.array(test_schedule_full_workplace)>0):
        all_probs = [calc_probs_at_least_one_full_workplace_testing(full_tests,
            extra_tests, proportion_infected, test_sensitivity, num_people) for
                     full_tests, extra_tests in zip(test_schedule_full_workplace, remaining_tests_schedule)]
    else:
        all_probs = [calc_prob_detect_at_least_one(test_schedule, proportion_infected, test_sensitivity) for
                     test_schedule in all_test_schedules]
    return all_probs, np.mean(all_probs), min(all_probs), max(all_probs)

def calc_probs_at_least_one_full_workplace_testing(test_schedule_full_workplace,
            remaining_tests_schedule, prevalence_proportion, sensitivity, num_people):
    prob_zero_each_day = \
        [binom.cdf(0, tests, prob * sensitivity) for
         tests, prob in zip(remaining_tests_schedule, prevalence_proportion)]
    prob_zero_each_day_full_coverage = (1-sensitivity)**(np.array(prevalence_proportion)*num_people*np.array(test_schedule_full_workplace))
    prob_all_zeros = np.prod(prob_zero_each_day)*np.prod(prob_zero_each_day_full_coverage)
    prob_detect = 1 - prob_all_zeros
    return prob_detect

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
    if number_of_times_testing_occurs_per_week == 6:
        test_days = (1, 1, 1, 1, 1, 1, 0)
    if number_of_times_testing_occurs_per_week == 7:
        test_days = (1, 1, 1, 1, 1, 1, 1)
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

    workplace_size_plot = False
    workplace_testing_frequency_plot = False
    test_sensitivity_plot = False
    reff_plot = True
    test_coverage_plot = False

    hq_plot = False
    base_sensitivity_frequency_full_coverage_plot = False
    variable_r0_plot = False

    num_tests = [0, 0, 5, 5]
    prev = [1 / 100] * 4

    # print(calc_prob_detect_at_least_one(num_tests, prev,.85))
    simple_exponential_growth(1, r_eff=1.5, num_days=10)

    test_pr = workplace_detection(num_people=10, growth_rate=1.5,
                                  number_of_times_testing_occurs_per_week=5,
                                  proportion_workplace_tested_per_week=.5, time_horizon=7)

    print(test_pr[1])

    base_num_people = 50 #workplace size
    base_growth_rate = 1.1 #reff
    base_number_test_times = 3 #test on mon/wed/fri
    base_prop_per_week = .5 #testing 50% of the workplace each week
    base_test_sensitivity = .85 #test sensitivity

    base_time_horizon = 14  # all the probabilities in the plots are the probability of detecting the outbreak within 14 days

    low_prob_per_week = .25
    high_prob_per_week = 1


    if base_sensitivity_frequency_full_coverage_plot:

        prop_list = [7, 3, 1]
        time_window_list = [7, 14]
        test_sensitivity_list = list(np.linspace(.4, .99, 30))
        fig, axs = plt.subplots(1, 2, figsize=(9, 4))

        for time_window, i in zip(time_window_list, count()):
            pr_list = []
            for test_proportion in prop_list:
                pr_sens_list = []
                for test_sens in test_sensitivity_list:
                    pr = workplace_detection(50, growth_rate=base_growth_rate,
                                             number_of_times_testing_occurs_per_week=test_proportion,
                                             proportion_workplace_tested_per_week=test_proportion,
                                             time_horizon=time_window,
                                             test_sensitivity=test_sens)
                    pr_sens_list.append(pr[1])
                pr_list.append(pr_sens_list)
            print(i)
            axs[i].plot(np.array(test_sensitivity_list), np.array(pr_list).transpose())
            axs[i].legend(['Daily testing', '3 times per week testing', 'Weekly testing'])
            axs[i].set_xlabel('Test sensitivity')
            axs[i].set_ylabel(f'Probability of detection within {time_window} days')
            axs[i].set_ylim([0.4, 1.05])
            # plt.title(f'{time_window} day time horizon, Reff = {reff}')
            # plt.savefig(f'Figures_workplace/HQ_sensitivity_{time_window}day_reff{reff}.png')

        plt.savefig('Figures_workplace/sensitivity_frequency_high_coverage_7_14_days.png')
        plt.show()

    if variable_r0_plot:

        test_proportion = 1
        time_window = 7
        test_sensitivity_list = list(np.linspace(.7, .99, 30))
        reff_list = [1.1, 1.5, 2, 2.5]
        for reff in reff_list:
            pr_list = []
            for test_sens in test_sensitivity_list:
                pr = workplace_detection(50, growth_rate=reff,
                                         number_of_times_testing_occurs_per_week=test_proportion,
                                         proportion_workplace_tested_per_week=test_proportion,
                                         time_horizon=time_window,
                                         test_sensitivity=test_sens)
                pr_list.append(pr[1])
            plt.plot(test_sensitivity_list, pr_list)
        plt.legend([f'Reff={i}' for i in reff_list])
        plt.ylabel('Probability of detection within 7 days')
        plt.xlabel('Test sensitivity')
        plt.title('Probability of detection with weekly testing')
        plt.savefig('Figures_workplace/prob_detect_varying_reff.png')

        plt.show()




    if hq_plot:
        pass
        workplace_detection(base_num_people, growth_rate=base_growth_rate,
                            number_of_times_testing_occurs_per_week=5,
                            proportion_workplace_tested_per_week=5,
                            time_horizon=14,
                            test_sensitivity=.6)[1]

    if test_coverage_plot:
        # proportion_list = [1, 3/4, 1/2, 1/3, 1/4, 1/6, 1/8]
        proportion_list = [i/8 for i in range(1, 9)]
        # proportion_list = list(np.linspace(.1,1,20))
        test_sensitivity_list = [.95, .85, .75, .65]
        for test_sensitivity in test_sensitivity_list:
            pr_list = []
            for test_proportion in proportion_list:
                pr = workplace_detection(base_num_people, growth_rate=base_growth_rate,
                                         number_of_times_testing_occurs_per_week=base_number_test_times,
                                         proportion_workplace_tested_per_week=test_proportion,
                                         time_horizon=base_time_horizon,
                                         test_sensitivity=test_sensitivity)
                pr_list.append(pr[1])
            plt.plot([i*100 for i in proportion_list], pr_list)
        plt.xlabel('Percentage of workplace tested each week')
        plt.ylabel('Probability of detection')
        plt.title('Proportion tested per week')
        plt.legend([f'Test sensitivity: {i}' for i in test_sensitivity_list])
        plt.savefig('Figures_workplace/workplace_prop_vary.png')
        plt.show()


    if reff_plot:
        reff_list = list(np.linspace(1, 2.5, 20))
        test_sensitivity_list = [.95, .85, .75, .65]
        for test_sensitivity in test_sensitivity_list:
            pr_list = []
            for ref in reff_list:
                pr = workplace_detection(base_num_people, growth_rate=ref,
                                         number_of_times_testing_occurs_per_week=base_number_test_times,
                                         proportion_workplace_tested_per_week=base_prop_per_week,
                                         time_horizon=base_time_horizon,
                                         test_sensitivity=test_sensitivity)
                pr_list.append(pr[1])
            plt.plot(reff_list, pr_list)
        plt.xlabel('Reff')
        plt.ylabel('Probability of detection')
        plt.title('Growth rate')
        plt.legend([f'Test sensitivity: {i}' for i in test_sensitivity_list])
        plt.savefig('Figures_workplace/workplace_reff_vary.png')
        plt.show()

    if test_sensitivity_plot:
        test_sensitivity_list = list(np.linspace(.5, .95, 10))
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

        # hotel quarantine
        test_sensitivity_list = list(np.linspace(.05, .99, 10))
        prop_list = [1, 3, 7]
        hq_growth = 1.5
        detection_window_list = [7, 14]
        reff_list = [1, 1.1, 1.5, 2]
        workplace_size_list = [50]
        for reff in reff_list:
            for workplace_size in workplace_size_list:
                for time_window in detection_window_list:
                    pr_list = []
                    for test_proportion in prop_list:
                        pr_sens_list = []
                        for test_sens in test_sensitivity_list:
                            pr = workplace_detection(workplace_size, growth_rate=reff,
                                                     number_of_times_testing_occurs_per_week=test_proportion,
                                                     proportion_workplace_tested_per_week=test_proportion,
                                                     time_horizon=time_window,
                                                     test_sensitivity=test_sens)
                            pr_sens_list.append(pr[1])
                        pr_list.append(pr_sens_list)
                    plt.plot(np.array(test_sensitivity_list), np.array(pr_list).transpose())
                    plt.legend(['Weekly testing', '3 times per week testing', 'Daily testing'])
                    plt.xlabel('Test sensitivity')
                    plt.ylabel(f'Probability of detection within {time_window} days')
                    plt.title(f'{time_window} day time horizon, Reff = {reff}')
                    plt.savefig(f'Figures_workplace/HQ_sensitivity_{time_window}day_reff{reff}.png')
                    plt.show()
                    plt.close()
    if workplace_testing_frequency_plot:
        test_freq_list = list(range(1,8))
        pr_list = []
        pr_list_low = []
        pr_list_high = []
        for test_freq in test_freq_list:
            pr = workplace_detection(base_num_people, growth_rate=base_growth_rate,
                                     number_of_times_testing_occurs_per_week=test_freq,
                                     proportion_workplace_tested_per_week=base_prop_per_week,
                                     time_horizon=base_time_horizon,
                                     test_sensitivity=base_test_sensitivity)
            pr_list.append(pr[1])
            pr_list_low.append(pr[2])
            pr_list_high.append(pr[3])
        plt.plot(test_freq_list, pr_list)
        plt.fill_between(test_freq_list, pr_list_low, pr_list_high, alpha=.25, color='b')

        plt.xticks(test_freq_list)
        plt.xlabel('Number of times testing occurs per week')
        plt.ylabel('Expected probability of detection')
        plt.title('Probability of detection, varying test frequency')
        plt.savefig('Figures_workplace/workplace_test_frequency.png')
        plt.show()

    if workplace_size_plot:
        min_pop, max_pop = 2, 50
        # pop_size = [2, 6, 10, 20, 30, 50, 100, 200]
        pop_size = list(range(min_pop, max_pop,2))
        pr_list = []
        for pop in pop_size:
            pr = workplace_detection(pop, growth_rate=base_growth_rate,
                                     number_of_times_testing_occurs_per_week=base_number_test_times,
                                     proportion_workplace_tested_per_week=base_prop_per_week,
                                     time_horizon=base_time_horizon,
                                     test_sensitivity=base_test_sensitivity)
            pr_list.append(pr[1])
        pr_min = workplace_detection(1000, growth_rate=base_growth_rate,
                                     number_of_times_testing_occurs_per_week=base_number_test_times,
                                     proportion_workplace_tested_per_week=base_prop_per_week,
                                     time_horizon=base_time_horizon,
                                     test_sensitivity=base_test_sensitivity)
        plt.plot(pop_size, pr_list)
        plt.plot([min_pop, max_pop], [pr_min[1], pr_min[1]], '--r')
        plt.xlabel('Workplace size')
        plt.ylabel('Probability of detection')
        plt.title('Probability of detection with varying workplace size')
        plt.savefig('Figures_workplace/workplace_size.png')
        plt.show()

    # prev_df = pd.DataFrame()
    # for reff in [1.1, 1.5, 2]:
    #     prev_df[f'Reff = {reff}'] = simple_exponential_growth(1,reff, 14)
    # prev_df.to_csv('Simple_prev_growth_table.csv')