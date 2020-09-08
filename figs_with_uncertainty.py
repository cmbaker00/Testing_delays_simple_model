import numpy as np
import matplotlib.pyplot as plt
from SimpleModelsModule import TestOptimisation
from Plot_all_scenarios import make_onward_transmission_vector, make_population_tuple
import Scenario_parameter_values as scenario

def sample_onward_transmission(onwards_dict):
    close_contact = np.random.uniform(*onwards_dict['close_contact'])
    symptomatic = np.random.uniform(*onwards_dict['symptomatic'])
    asymptomatic = np.random.uniform(*onwards_dict['asymptomatic'])
    return make_onward_transmission_vector(close_contact=close_contact,
                                    symptomatic=symptomatic,
                                    asymptomatic=asymptomatic)

def sample_prob_indication(prob_dict):
    close_contact = np.random.uniform(*prob_dict['close_contact'])
    symptomatic = np.random.uniform(*prob_dict['symptomatic'])
    asymptomatic = np.random.uniform(*prob_dict['asymptomatic'])
    return close_contact, symptomatic, asymptomatic


def sample_population(pop_dict, prob_indication):
    close_contact = np.random.uniform(*pop_dict['close_contact'])
    symp = np.random.uniform(*pop_dict['symptomatic'])
    total_pop = pop_dict['total_population']
    pop, exp_cases = make_population_tuple(num_close=close_contact,
                                           num_symp=symp,
                                           total_pop=total_pop,
                                           presenting_proporition=1,
                                           probability_by_indication=prob_indication)
    return pop

def sample_capacitiy(cap_range):
    rand_cap = np.random.uniform(*cap_range)
    rounded_capacity = 10*int(rand_cap/10)
    return rounded_capacity

def sample_present_prop(pres_range):
    return np.random.uniform(*pres_range)

def run_analysis_save_plot(priority, onward_transmission, pop, pre_prob, cap, prop_symp, reps, scenario_name,
                           plot_title=None, test_order=None):
    onward_transmission = {key: value * 2 if len(value) == 1 else value for key, value in onward_transmission.items()}
    pop_new = {}
    for key, value in pop.items():
        if key == 'total_population':
            pop_new[key] = value
        else:
            pop_new[key] = value*2 if len(value) == 1 else value
    pop = pop_new
    pre_prob = {key: value * 2 if len(value) == 1 else value for key, value in pre_prob.items()}
    cap = cap*2 if len(cap)==1 else cap
    prop_symp = prop_symp*2 if len(prop_symp)==1 else prop_symp

    onward_transmission_store = []
    for i in range(reps):
        print(i)
        current_onward = sample_onward_transmission(onward_transmission)
        current_prob = sample_prob_indication(pre_prob)
        current_pop = sample_population(pop, current_prob)
        current_cap = sample_capacitiy(cap)
        current_pres = sample_present_prop(prop_symp)

        test_optim = TestOptimisation(priority_queue=priority, onward_transmission=current_onward,
                                      population=current_pop,
                                      pre_test_probability=current_prob,
                                      routine_capacity=current_cap,
                                      symptomatic_testing_proportion=current_pres,
                                      test_prioritsation_by_indication=test_order)

        max_test = 1500
        max_test_prop = max_test/current_cap
        test_array, onward_transmission_array, positivity = test_optim.generate_onward_transmission_with_tests(
            max_tests_proportion=max_test_prop)

        onward_transmission_array = 100 * onward_transmission_array / max(onward_transmission_array) #max onward transmission shouldn not depend on symp perc.
        test_array = np.array(test_array) / 100

        onward_transmission_store.append(list(onward_transmission_array))

        # plt.plot(test_array, onward_transmission_array)
        # plt.show()
    onward_transmission_full_array = np.array(onward_transmission_store)
    median = np.percentile(onward_transmission_full_array, 50, axis=0)
    low_ci = np.percentile(onward_transmission_full_array, 5, axis=0)
    up_ci  = np.percentile(onward_transmission_full_array, 95, axis=0)
    low_ci2 = np.percentile(onward_transmission_full_array, 15, axis=0)
    up_ci2  = np.percentile(onward_transmission_full_array, 85, axis=0)
    low_ci3 = np.percentile(onward_transmission_full_array, 25, axis=0)
    up_ci3  = np.percentile(onward_transmission_full_array, 75, axis=0)
    low_ci4 = np.percentile(onward_transmission_full_array, 35, axis=0)
    up_ci4  = np.percentile(onward_transmission_full_array, 65, axis=0)
    low_ci5 = np.percentile(onward_transmission_full_array, 45, axis=0)
    up_ci5  = np.percentile(onward_transmission_full_array, 55, axis=0)
    plt.plot(test_array, median, 'k')
    plt.fill_between(test_array, low_ci, up_ci, alpha=.25, color='b')
    plt.fill_between(test_array, low_ci2, up_ci2, alpha=.25, color='b')
    plt.fill_between(test_array, low_ci3, up_ci3, alpha=.25, color='b')
    plt.fill_between(test_array, low_ci4, up_ci4, alpha=.25, color='b')
    plt.fill_between(test_array, low_ci5, up_ci5, alpha=.25, color='b')
    # plt.plot(test_array, low_ci)
    # plt.plot(test_array, up_ci)
    plt.xlabel('Tests per 1000')
    plt.ylim((65, 100))
    plt.ylabel('Percentage of onwards transmission')
    if plot_title:
        plt.title(plot_title)
    plt.savefig(f'Figures_uncertainty/{scenario_name}.png')
    plt.close()
    print(f'{scenario_name} -- Done!')


#
#
#
# priority_queue = True
# onward_transmission_range = {'close_contact': [0.25, 1],
#                              'symptomatic': [1, 1.5],
#                              'asymptomatic': [.5, 1.5]}
# pop_distribution_range = {'close_contact': [10, 100],
#                           'symptomatic': [400, 800],
#                           'total_population': 100000}
# pre_prob_range = {'close_contact': [0.01, 0.05],
#                   'symptomatic': [0.001, 0.01],
#                   'asymptomatic': [0.00001, 0.0001]}
# prop_symp_range = [.5, .95]
# cap_range = [400, 400]
#
# run_analysis_save_plot(priority=priority_queue,
#                        onward_transmission=onward_transmission_range,
#                        pop=pop_distribution_range,
#                        pre_prob=pre_prob_range,
#                        cap=cap_range,
#                        prop_symp=prop_symp_range,
#                        reps=100,
#                        scenario_name='uncertainty_test')
#
cc_on, symp_on, asymp_on = scenario.onward_transmission_high
cc_prob, symp_prob, asymp_prob = scenario.test_prob_high
cc_pop, symp_pop = scenario.pop_high
total_pop = scenario.total_population


cc_on_outbreak, symp_on_outbreak, asymp_on_outbreak = scenario.onward_transmission_low
cc_prob_outbreak, symp_prob_outbreak, asymp_prob_outbreak = scenario.test_prob_low
cc_pop_outbreak, symp_pop_outbreak = scenario.pop_low
total_pop_outbreak = scenario.total_population

# # NO VARIATION TEMPLATE START
# priority_queue = True
# onward_transmission_range = {'close_contact': [cc_on],
#                              'symptomatic': [symp_on],
#                              'asymptomatic': [asymp_on]}
# pop_distribution_range = {'close_contact': [cc_pop],
#                           'symptomatic': [symp_pop],
#                           'total_population': total_pop}
# pre_prob_range = {'close_contact': [cc_prob],
#                              'symptomatic': [symp_prob],
#                              'asymptomatic': [asymp_prob]}
#
# prop_symp_range = [.5]
# cap_range = [scenarios.test_capacity_high]
#
# run_analysis_save_plot(priority=priority_queue,
#                        onward_transmission=onward_transmission_range,
#                        pop=pop_distribution_range,
#                        pre_prob=pre_prob_range,
#                        cap=cap_range,
#                        prop_symp=prop_symp_range,
#                        reps=100,
#                        scenario_name='presenting_prop_range_25_75',
#                        plot_title=None)
# # NO VARIATION TEMPLATE END

run_symp_presentation_range = False
run_pre_test_prob_range = True
run_pop_distribution_range = True
run_onward_transmission_range = True
run_test_number_uncertainty = False

if run_symp_presentation_range:
    priority_queue = True
    onward_transmission_range = {'close_contact': [cc_on],
                                 'symptomatic': [symp_on],
                                 'asymptomatic': [asymp_on]}
    pop_distribution_range = {'close_contact': [cc_pop],
                              'symptomatic': [symp_pop],
                              'total_population': total_pop}
    pre_prob_range = {'close_contact': [cc_prob],
                                 'symptomatic': [symp_prob],
                                 'asymptomatic': [asymp_prob]}

    prop_symp_range = [.25, .75]


    cap_range = [scenario.test_capacity_high]

    run_analysis_save_plot(priority=priority_queue,
                           onward_transmission=onward_transmission_range,
                           pop=pop_distribution_range,
                           pre_prob=pre_prob_range,
                           cap=cap_range,
                           prop_symp=prop_symp_range,
                           reps=100,
                           scenario_name='presenting_prop_range_25_75_community_transmission_capacity_high',
                           plot_title='Presenting proportion between 25% and 75%',
                           test_order=scenario.priority_order[0])


    run_analysis_save_plot(priority=priority_queue,
                           onward_transmission=onward_transmission_range,
                           pop=pop_distribution_range,
                           pre_prob=pre_prob_range,
                           cap=cap_range,
                           prop_symp=prop_symp_range,
                           reps=100,
                           scenario_name='presenting_prop_range_25_75_community_transmission_capacity_high_symp_priority',
                           plot_title='Presenting proportion between 25% and 75%',
                           test_order=scenario.priority_order[1])


    cap_range = [scenario.test_capacity_low]

    run_analysis_save_plot(priority=priority_queue,
                           onward_transmission=onward_transmission_range,
                           pop=pop_distribution_range,
                           pre_prob=pre_prob_range,
                           cap=cap_range,
                           prop_symp=prop_symp_range,
                           reps=100,
                           scenario_name='presenting_prop_range_25_75_community_transmission_capacity_low',
                           plot_title='Presenting proportion between 25% and 75%',
                           test_order=scenario.priority_order[0])


    run_analysis_save_plot(priority=priority_queue,
                           onward_transmission=onward_transmission_range,
                           pop=pop_distribution_range,
                           pre_prob=pre_prob_range,
                           cap=cap_range,
                           prop_symp=prop_symp_range,
                           reps=100,
                           scenario_name='presenting_prop_range_25_75_community_transmission_capacity_low_symp_priority',
                           plot_title='Presenting proportion between 25% and 75%',
                           test_order=scenario.priority_order[1])

    priority_queue = True
    onward_transmission_range = {'close_contact': [cc_on_outbreak],
                                 'symptomatic': [symp_on_outbreak],
                                 'asymptomatic': [asymp_on_outbreak]}
    pop_distribution_range = {'close_contact': [cc_pop_outbreak],
                              'symptomatic': [symp_pop_outbreak],
                              'total_population': total_pop_outbreak}
    pre_prob_range = {'close_contact': [cc_prob_outbreak],
                      'symptomatic': [symp_prob_outbreak],
                      'asymptomatic': [asymp_prob_outbreak]}

    prop_symp_range = [.25, .75]

    cap_range = [scenario.test_capacity_high]

    run_analysis_save_plot(priority=priority_queue,
                           onward_transmission=onward_transmission_range,
                           pop=pop_distribution_range,
                           pre_prob=pre_prob_range,
                           cap=cap_range,
                           prop_symp=prop_symp_range,
                           reps=100,
                           scenario_name='presenting_prop_range_25_75_outbreak_response_capacity_high',
                           plot_title='Presenting proportion between 25% and 75%',
                           test_order=scenario.priority_order[0])

    run_analysis_save_plot(priority=priority_queue,
                           onward_transmission=onward_transmission_range,
                           pop=pop_distribution_range,
                           pre_prob=pre_prob_range,
                           cap=cap_range,
                           prop_symp=prop_symp_range,
                           reps=100,
                           scenario_name='presenting_prop_range_25_75_outbreak_response_capacity_high_symp_priority',
                           plot_title='Presenting proportion between 25% and 75%',
                           test_order=scenario.priority_order[1])

    cap_range = [scenario.test_capacity_low]

    run_analysis_save_plot(priority=priority_queue,
                           onward_transmission=onward_transmission_range,
                           pop=pop_distribution_range,
                           pre_prob=pre_prob_range,
                           cap=cap_range,
                           prop_symp=prop_symp_range,
                           reps=100,
                           scenario_name='presenting_prop_range_25_75_outbreak_response_capacity_low',
                           plot_title='Presenting proportion between 25% and 75%',
                           test_order=scenario.priority_order[0])

    run_analysis_save_plot(priority=priority_queue,
                           onward_transmission=onward_transmission_range,
                           pop=pop_distribution_range,
                           pre_prob=pre_prob_range,
                           cap=cap_range,
                           prop_symp=prop_symp_range,
                           reps=100,
                           scenario_name='presenting_prop_range_25_75_outbreak_response_capacity_low_symp_priority',
                           plot_title='Presenting proportion between 25% and 75%',
                           test_order=scenario.priority_order[1])

if run_onward_transmission_range:
    priority_queue = True
    onward_transmission_range = {'close_contact': [cc_on*.8, cc_on*1.2],
                                 'symptomatic': [symp_on*.8, symp_on*1.2],
                                 'asymptomatic': [asymp_on*.8, asymp_on*1.2]}
    pop_distribution_range = {'close_contact': [cc_pop],
                              'symptomatic': [symp_pop],
                              'total_population': total_pop}
    pre_prob_range = {'close_contact': [cc_prob],
                                 'symptomatic': [symp_prob],
                                 'asymptomatic': [asymp_prob]}

    prop_symp_range = [.5]
    cap_range = [scenario.test_capacity_high]

    run_analysis_save_plot(priority=priority_queue,
                           onward_transmission=onward_transmission_range,
                           pop=pop_distribution_range,
                           pre_prob=pre_prob_range,
                           cap=cap_range,
                           prop_symp=prop_symp_range,
                           reps=100,
                           scenario_name='onward_transmission_uncertainty',
                           plot_title='Onward transmission +/- 20%')

if run_pop_distribution_range:
    priority_queue = True
    onward_transmission_range = {'close_contact': [cc_on],
                                 'symptomatic': [symp_on],
                                 'asymptomatic': [asymp_on]}
    pop_distribution_range = {'close_contact': [cc_pop*.8, cc_pop*1.2],
                              'symptomatic': [symp_pop*.8, symp_pop*1.2],
                              'total_population': total_pop}
    pre_prob_range = {'close_contact': [cc_prob],
                                 'symptomatic': [symp_prob],
                                 'asymptomatic': [asymp_prob]}

    prop_symp_range = [.5]
    cap_range = [scenario.test_capacity_high]

    run_analysis_save_plot(priority=priority_queue,
                           onward_transmission=onward_transmission_range,
                           pop=pop_distribution_range,
                           pre_prob=pre_prob_range,
                           cap=cap_range,
                           prop_symp=prop_symp_range,
                           reps=100,
                           scenario_name='pop_distribution_uncertainty',
                           plot_title='Close contact and symptomatic population +/- 20%')


if run_pre_test_prob_range:
    priority_queue = True
    onward_transmission_range = {'close_contact': [cc_on],
                                 'symptomatic': [symp_on],
                                 'asymptomatic': [asymp_on]}
    pop_distribution_range = {'close_contact': [cc_pop],
                              'symptomatic': [symp_pop],
                              'total_population': total_pop}
    pre_prob_range = {'close_contact': [cc_prob*.8, cc_prob*1.2],
                                 'symptomatic': [symp_prob*.8, symp_prob*1.2],
                                 'asymptomatic': [asymp_prob*.8, asymp_prob*1.2]}

    prop_symp_range = [.5]
    cap_range = [scenario.test_capacity_high]

    run_analysis_save_plot(priority=priority_queue,
                           onward_transmission=onward_transmission_range,
                           pop=pop_distribution_range,
                           pre_prob=pre_prob_range,
                           cap=cap_range,
                           prop_symp=prop_symp_range,
                           reps=100,
                           scenario_name='test_prob_uncertainty',
                           plot_title='Pre-test probability of positive +/- 20%')

if run_test_number_uncertainty:
    priority = True

    total_population = scenario.total_population

    # High prevelance
    onward_transmission_vector_high = \
        make_onward_transmission_vector(*scenario.onward_transmission_high)

    test_prob_high = scenario.test_prob_high

    population_high, cases_high = \
        make_population_tuple(num_close=scenario.pop_high[0],
                              num_symp=scenario.pop_high[1],
                              total_pop=total_population,
                              presenting_proporition=1,
                              probability_by_indication=test_prob_high)

    test_optim = TestOptimisation(priority_queue=priority,
                                  onward_transmission=onward_transmission_vector_high,
                                  population=population_high,
                                  pre_test_probability=test_prob_high,
                                  routine_capacity=400,
                                  symptomatic_testing_proportion=1,
                                  test_prioritsation_by_indication=None)

    value_gap = .01
    uncertainty_range_values = [0] + \
                               list(np.arange(0,.5,value_gap) +
                                    value_gap)
    num_tests_list = []
    transmission_list = []

    max_num_tests_for_plot = np.inf

    for uncertainty_value in uncertainty_range_values:
        print(f"Running uncertainy value {uncertainty_value}")
        if uncertainty_value == 0:
            num_test_array, transmission, positivity = test_optim.generate_onward_transmission_with_tests()
            num_tests_list.append(np.array(num_test_array))
            transmission_list.append(np.array(transmission))
        else:
            num_tests, exp_transmission = test_optim.create_uncertain_onward_array(uncertainty_range_prop=
                                                                                   uncertainty_value)
            num_tests = np.array(num_tests)

            exp_transmission = np.array(exp_transmission)
            exp_transmission = exp_transmission/max(exp_transmission)

            num_tests_list.append(num_tests)
            transmission_list.append(exp_transmission)

            if max(num_tests) < max_num_tests_for_plot:
                max_num_tests_for_plot = max(num_tests)
    plot_tests = []
    plot_transmission = []
    for current_tests, current_exp_transmission in zip(num_tests_list, transmission_list):
        index_values = current_tests <= max_num_tests_for_plot
        # current_tests = current_tests[index_values]
        # current_exp_transmission = current_exp_transmission[index_values]
        plt_tests_values = current_tests[index_values]/100
        plt_transmission_values = current_exp_transmission[index_values]/\
                                  max(current_exp_transmission[index_values])
        plot_tests.append(plt_tests_values)
        plot_transmission.append(plt_transmission_values)
        plt.plot(plt_tests_values, plt_transmission_values,
                 color=[0, 0, 0],
                 alpha=.2)


    plt.xlabel('Tests per 1000')
    plt.ylabel('Percentage of onwards transmission')
    plt.savefig('Figures_uncertainty/test_number_uncertainty.png')
    plt.show()


