import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SimpleModelsModule import TestOptimisation
import Param_values_MS as scenario
import Plot_all_scenarios
import itertools

if __name__ == "__main__":
    test_figure_area = False
    tat_figure = True
    kretzschmar_figure = False
    onward_transmission_double_figure = False
    track_trace_impact_figure = False
    positive_percent_impact_figure = False
    supplement_pos_perc_figures = False
    supplement_figure_non_quadratic = False

    base_figure_directory = 'MS_figures'

    if onward_transmission_double_figure:
        total_population = scenario.total_population

        # High prevelance
        onward_transmission_vector_high = \
            Plot_all_scenarios.make_onward_transmission_vector(*scenario.onward_transmission_high)

        test_prob_high = scenario.test_prob_high

        population_high, cases_high = \
            Plot_all_scenarios.make_population_tuple(num_close=scenario.pop_high[0],
                                  num_symp=scenario.pop_high[1],
                                  total_pop=total_population,
                                  presenting_proporition=1,
                                  probability_by_indication=test_prob_high)

        print(f'Daily infections = {cases_high}')

        # Low prevelance
        onward_transmission_vector_low = \
            Plot_all_scenarios.make_onward_transmission_vector(*scenario.onward_transmission_low)

        test_prob_low = scenario.test_prob_low

        population_low, cases_low = \
            Plot_all_scenarios.make_population_tuple(num_close=scenario.pop_low[0],
                                  num_symp=scenario.pop_low[1],
                                  total_pop=total_population,
                                  presenting_proporition=1,
                                  probability_by_indication=test_prob_low)

        print(f'Daily infections = {cases_low}')

        # priority_values = [True, False]
        priority_values = [False]
        capacity_values = [scenario.test_capacity_low, scenario.test_capacity_high]
        # symp_prop_values = [.5, 1]
        symp_prop_values = [.5]
        scenario_names = ['Low_prev', 'High_prev']
        situation_dict = {'Low_prev': {'onward': onward_transmission_vector_low,
                                       'pop': population_low,
                                       'pre_prob': test_prob_low},
                          'High_prev': {'onward': onward_transmission_vector_high,
                                       'pop': population_high,
                                       'pre_prob': test_prob_high}
                          }
        priority_allocation_options = scenario.priority_order

        fig, axs = plt.subplots(1, 2, figsize=(8,4))
        axs_count = itertools.count()
        for priority_value in priority_values:
            for priority_order in priority_allocation_options:
                for capacity_value in capacity_values:
                    for symp_prop_value in symp_prop_values:

                        axs_current = next(axs_count)
                        for scenario in scenario_names:
                            c_dict = situation_dict[scenario]

                            test_optim = TestOptimisation(priority_queue=priority_value,
                                                          onward_transmission=c_dict['onward'],
                                                          population=c_dict['pop'],
                                                          pre_test_probability=c_dict['pre_prob'],
                                                          routine_capacity=capacity_value,
                                                          symptomatic_testing_proportion=symp_prop_value,
                                                          test_prioritsation_by_indication=priority_order)
                            test_array, transmission_array, positive_array = \
                                test_optim.generate_onward_transmission_with_tests(max_tests_proportion=1000/capacity_value)


                            axs[axs_current].plot([i/100 for i in test_array], 100*transmission_array/max(transmission_array))

                        axs[axs_current].set(xlabel='Number of tests per 1000')
                        if axs_current == 0:
                            axs[axs_current].set(ylabel='Percentage of onward transmission')
                        axs[axs_current].legend(['Outbreak response', 'Community transmission'])
                        # axs[axs_current].set
                        axs[axs_current].plot([capacity_value/100]*2, [75, 100],'--r')
        plt.savefig(f'{base_figure_directory}/Onward_transmission_two_panel.png')
        plt.show()
        print(1)
                            # Plot_all_scenarios.run_analysis_save_plot(priority=priority_value,
                            #                                           onward_transmission=c_dict['onward'],
                            #                                           pop=c_dict['pop'],
                            #                                           pre_prob=c_dict['pre_prob'],
                            #                                           cap=capacity_value,
                            #                                           prop_symp=symp_prop_value,
                            #                                           scenario_name=scenario,
                            #                                           priority_ordering=priority_order,
                            #                                           directory_name=base_figure_directory)


    if track_trace_impact_figure:
        total_population = scenario.total_population

        # High prevelance
        onward_transmission_vector_high = \
            Plot_all_scenarios.make_onward_transmission_vector(*scenario.onward_transmission_high)

        test_prob_high = scenario.test_prob_high

        population_high, cases_high = \
            Plot_all_scenarios.make_population_tuple(num_close=scenario.pop_high[0],
                                  num_symp=scenario.pop_high[1],
                                  total_pop=total_population,
                                  presenting_proporition=1,
                                  probability_by_indication=test_prob_high)

        print(f'Daily infections = {cases_high}')

        # Low prevelance
        onward_transmission_vector_low = \
            Plot_all_scenarios.make_onward_transmission_vector(*scenario.onward_transmission_low)

        test_prob_low = scenario.test_prob_low

        population_low, cases_low = \
            Plot_all_scenarios.make_population_tuple(num_close=scenario.pop_low[0],
                                  num_symp=scenario.pop_low[1],
                                  total_pop=total_population,
                                  presenting_proporition=1,
                                  probability_by_indication=test_prob_low)

        print(f'Daily infections = {cases_low}')

        # priority_values = [True, False]
        priority_values = [False]
        capacity_values = [scenario.test_capacity_low, scenario.test_capacity_high]
        # symp_prop_values = [.5, 1]
        symp_prop_values = [.5]
        scenario_names = ['Low_prev']
        situation_dict = {'Low_prev': {'onward': onward_transmission_vector_low,
                                       'pop': population_low,
                                       'pre_prob': test_prob_low},
                          'High_prev': {'onward': onward_transmission_vector_high,
                                       'pop': population_high,
                                       'pre_prob': test_prob_high}
                          }
        priority_allocation_options = scenario.priority_order

        for priority_value in priority_values:
            for priority_order in priority_allocation_options:
                for capacity_value in capacity_values:
                    for symp_prop_value in symp_prop_values:

                        for scenario in scenario_names:
                            c_dict = situation_dict[scenario]

                            test_optim = TestOptimisation(priority_queue=priority_value,
                                                          onward_transmission=c_dict['onward'],
                                                          population=c_dict['pop'],
                                                          pre_test_probability=c_dict['pre_prob'],
                                                          routine_capacity=capacity_value,
                                                          symptomatic_testing_proportion=symp_prop_value,
                                                          test_prioritsation_by_indication=priority_order)
                            test_optim_iso_only = TestOptimisation(priority_queue=priority_value,
                                                                   onward_transmission=c_dict['onward'],
                                                                   population=c_dict['pop'],
                                                                   pre_test_probability=c_dict['pre_prob'],
                                                                   routine_capacity=capacity_value,
                                                                   symptomatic_testing_proportion=symp_prop_value,
                                                                   test_prioritsation_by_indication=priority_order,
                                                                   routine_tat=10,
                                                                   tat_at_fifty_percent_surge=20
                                                                   )
                            test_array, transmission_array, positive_array = \
                                test_optim.generate_onward_transmission_with_tests(max_tests_proportion=1000/capacity_value)

                            test_array_iso, transmission_array_iso, positive_array_iso = \
                                test_optim_iso_only.generate_onward_transmission_with_tests(max_tests_proportion=1000/capacity_value)

                            no_test_onward_transmission = test_optim.estimate_transmission_with_testing(0)[0]

                            transmission_reduction = no_test_onward_transmission - transmission_array
                            iso_reduction = no_test_onward_transmission - transmission_array_iso
                            perc_trace = (1 - iso_reduction/transmission_reduction)*100
                            # transmission_reduction = max(transmission_array) - transmission_array
                            # transmission_reduction_iso = max(transmission_array_iso) - transmission_array_iso
                            transmisison_difference = transmission_array_iso-transmission_array
                            percentage_transmission_difference = transmisison_difference/transmission_array_iso
                            # print(min(transmisison_difference))
                            plt.plot([i/100 for i in test_array], perc_trace)

                            # axs[axs_current].plot([i/100 for i in test_array], transmission_reduction_iso)

                        plt.xlabel('Number of tests per 1000')
                        plt.ylabel('Percentage reduction through contact tracing')
                        plt.legend(['Routine capacity 2 per 1000', 'Routine capacity 4 per 1000'])
                        # axs[axs_current].set
                        # axs[axs_current].plot([capacity_value/100]*2, [0, 45],'--r')
        plt.savefig(f'{base_figure_directory}/Percenage_reduction_by_tracing.png')
        plt.show()
        print(1)
                            # Plot_all_scenarios.run_analysis_save_plot(priority=priority_value,
                            #                                           onward_transmission=c_dict['onward'],
                            #                                           pop=c_dict['pop'],
                            #                                           pre_prob=c_dict['pre_prob'],
                            #                                           cap=capacity_value,
                            #                                           prop_symp=symp_prop_value,
                            #                                           scenario_name=scenario,
                            #                                           priority_ordering=priority_order,
                            #                                           directory_name=base_figure_directory)

    if positive_percent_impact_figure:
        total_population = scenario.total_population

        # High prevelance
        onward_transmission_vector_high = \
            Plot_all_scenarios.make_onward_transmission_vector(*scenario.onward_transmission_high)

        test_prob_high = scenario.test_prob_high

        population_high, cases_high = \
            Plot_all_scenarios.make_population_tuple(num_close=scenario.pop_high[0],
                                  num_symp=scenario.pop_high[1],
                                  total_pop=total_population,
                                  presenting_proporition=1,
                                  probability_by_indication=test_prob_high)

        print(f'Daily infections = {cases_high}')

        # Low prevelance
        onward_transmission_vector_low = \
            Plot_all_scenarios.make_onward_transmission_vector(*scenario.onward_transmission_low)

        test_prob_low = scenario.test_prob_low

        population_low, cases_low = \
            Plot_all_scenarios.make_population_tuple(num_close=scenario.pop_low[0],
                                  num_symp=scenario.pop_low[1],
                                  total_pop=total_population,
                                  presenting_proporition=1,
                                  probability_by_indication=test_prob_low)

        print(f'Daily infections = {cases_low}')

        # priority_values = [True, False]
        priority_values = [False]
        capacity_values = [scenario.test_capacity_low, scenario.test_capacity_high]
        # symp_prop_values = [.5, 1]
        symp_prop_values = [.5]
        scenario_names = ['Low_prev', 'High_prev']
        situation_dict = {'Low_prev': {'onward': onward_transmission_vector_low,
                                       'pop': population_low,
                                       'pre_prob': test_prob_low},
                          'High_prev': {'onward': onward_transmission_vector_high,
                                       'pop': population_high,
                                       'pre_prob': test_prob_high}
                          }
        priority_allocation_options = scenario.priority_order

        priority_value = priority_values[0]
        priority_order = priority_allocation_options[0]
        capacity_value = capacity_values[0]
        symp_prop_value = symp_prop_values[0]
        scenario = scenario_names[1]

        c_dict = situation_dict[scenario]

        test_optim = TestOptimisation(priority_queue=priority_value,
                                      onward_transmission=c_dict['onward'],
                                      population=c_dict['pop'],
                                      pre_test_probability=c_dict['pre_prob'],
                                      routine_capacity=capacity_value,
                                      symptomatic_testing_proportion=symp_prop_value,
                                      test_prioritsation_by_indication=priority_order)

        test_array, transmission_array, positive_array = \
            test_optim.generate_onward_transmission_with_tests(max_tests_proportion=1000/capacity_value)
        test_nparray = np.array(test_array)/100
        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        color = [0,0,.6]
        ax1.plot(test_nparray, 100*transmission_array/max(transmission_array), color=color)
        ax1.set_ylabel('Percentage of onwards transmission', color=color)
        ax1.set_xlabel('Number of tests per 1000')

        min_transmission_tests = test_nparray[np.where(transmission_array==min(transmission_array))]
        ax1.plot([min_transmission_tests]*2, [80,100],'k--')

        ax2 = ax1.twinx()
        color = [.25, .4, 0]
        ax2.plot([i/100 for i in test_array], 100*positive_array*np.array(test_array)/cases_high, color=color)
        ax2.set_ylabel('Percentage of infections identified', color=color)

        fig.tight_layout()
        # plt.xlabel('Number of tests per 1000')
        # plt.ylabel('Percentage reduction through contact tracing')
        # plt.legend(['Routine capacity 2 per 1000', 'Routine capacity 4 per 1000'])
        # axs[axs_current].set
        # axs[axs_current].plot([capacity_value/100]*2, [0, 45],'--r')
        plt.savefig(f'{base_figure_directory}/Cases_identified_transmission.png')
        plt.show()
        print(1)
                            # Plot_all_scenarios.run_analysis_save_plot(priority=priority_value,
                            #                                           onward_transmission=c_dict['onward'],
                            #                                           pop=c_dict['pop'],
                            #                                           pre_prob=c_dict['pre_prob'],
                            #                                           cap=capacity_value,
                            #                                           prop_symp=symp_prop_value,
                            #                                           scenario_name=scenario,
                            #                                           priority_ordering=priority_order,
                            #                                           directory_name=base_figure_directory)

    if supplement_pos_perc_figures:
        total_population = scenario.total_population

        # High prevelance
        onward_transmission_vector_high = \
            Plot_all_scenarios.make_onward_transmission_vector(*scenario.onward_transmission_high)

        test_prob_high = scenario.test_prob_high

        population_high, cases_high = \
            Plot_all_scenarios.make_population_tuple(num_close=scenario.pop_high[0],
                                  num_symp=scenario.pop_high[1],
                                  total_pop=total_population,
                                  presenting_proporition=1,
                                  probability_by_indication=test_prob_high)

        print(f'Daily infections = {cases_high}')

        # Low prevelance
        onward_transmission_vector_low = \
            Plot_all_scenarios.make_onward_transmission_vector(*scenario.onward_transmission_low)

        test_prob_low = scenario.test_prob_low

        population_low, cases_low = \
            Plot_all_scenarios.make_population_tuple(num_close=scenario.pop_low[0],
                                  num_symp=scenario.pop_low[1],
                                  total_pop=total_population,
                                  presenting_proporition=1,
                                  probability_by_indication=test_prob_low)

        print(f'Daily infections = {cases_low}')

        # priority_values = [True, False]
        priority_values = [False]
        capacity_values = [scenario.test_capacity_low, scenario.test_capacity_high]
        # symp_prop_values = [.5, 1]
        symp_prop_values = [.25, .5, .75]
        scenario_names = ['Low_prev', 'High_prev']
        situation_dict = {'Low_prev': {'onward': onward_transmission_vector_low,
                                       'pop': population_low,
                                       'pre_prob': test_prob_low},
                          'High_prev': {'onward': onward_transmission_vector_high,
                                       'pop': population_high,
                                       'pre_prob': test_prob_high}
                          }
        priority_allocation_options = scenario.priority_order

        priority_value = priority_values[0]
        priority_order = priority_allocation_options[0]
        for capacity_value in capacity_values:
            for symp_prop_value in symp_prop_values:
                for scenario in scenario_names:
                    c_dict = situation_dict[scenario]
                    test_optim = TestOptimisation(priority_queue=priority_value,
                                                  onward_transmission=c_dict['onward'],
                                                  population=c_dict['pop'],
                                                  pre_test_probability=c_dict['pre_prob'],
                                                  routine_capacity=capacity_value,
                                                  symptomatic_testing_proportion=symp_prop_value,
                                                  test_prioritsation_by_indication=priority_order)

                    test_array, transmission_array, positive_array = \
                        test_optim.generate_onward_transmission_with_tests(max_tests_proportion=1000/capacity_value)
                    test_nparray = np.array(test_array)/100
                    fig, ax1 = plt.subplots()
                    scenario_plot_name = None
                    if scenario == 'Low_prev':
                        scenario_plot_name = 'Outbreak response'
                    if scenario == 'High_prev':
                        scenario_plot_name = 'Community transmission'
                    plt.title(f'{scenario_plot_name} scenario\ntest capacity = {capacity_value/100} per 1000, symptomatic presenting proportion = {symp_prop_value}')
                    color = 'tab:blue'
                    color = [0,0,.6]
                    ax1.plot(test_nparray, 100*transmission_array/max(transmission_array), color=color)
                    ax1.set_ylabel('Percentage of onwards transmission', color=color)
                    ax1.set_xlabel('Number of tests per 1000')

                    # min_transmission_tests = test_nparray[np.where(transmission_array==min(transmission_array))]
                    # ax1.plot([min_transmission_tests]*2, [80,100],'k--')

                    ax2 = ax1.twinx()
                    color = [.25, .4, 0]
                    ax2.plot([i/100 for i in test_array], 100*positive_array*np.array(test_array)/cases_high, color=color)
                    ax2.set_ylabel('Percentage of infections identified', color=color)

                    fig.tight_layout()
                    # plt.xlabel('Number of tests per 1000')
                    # plt.ylabel('Percentage reduction through contact tracing')
                    # plt.legend(['Routine capacity 2 per 1000', 'Routine capacity 4 per 1000'])
                    # axs[axs_current].set
                    # axs[axs_current].plot([capacity_value/100]*2, [0, 45],'--r')
                    plt.savefig(f'{base_figure_directory}/Supplement_figures/Cases_identified_transmission_{scenario}_capacity{capacity_value}_sympprop_{symp_prop_value}.png')
                    # plt.show()


    if supplement_figure_non_quadratic:
        total_population = scenario.total_population

        # High prevelance
        onward_transmission_vector_high = \
            Plot_all_scenarios.make_onward_transmission_vector(*scenario.onward_transmission_high)

        test_prob_high = scenario.test_prob_high

        population_high, cases_high = \
            Plot_all_scenarios.make_population_tuple(num_close=scenario.pop_high[0],
                                  num_symp=scenario.pop_high[1],
                                  total_pop=total_population,
                                  presenting_proporition=1,
                                  probability_by_indication=test_prob_high)

        print(f'Daily infections = {cases_high}')

        # Low prevelance
        onward_transmission_vector_low = \
            Plot_all_scenarios.make_onward_transmission_vector(*scenario.onward_transmission_low)

        test_prob_low = scenario.test_prob_low

        population_low, cases_low = \
            Plot_all_scenarios.make_population_tuple(num_close=scenario.pop_low[0],
                                  num_symp=scenario.pop_low[1],
                                  total_pop=total_population,
                                  presenting_proporition=1,
                                  probability_by_indication=test_prob_low)

        print(f'Daily infections = {cases_low}')

        # priority_values = [True, False]
        priority_values = [False]
        capacity_values = [scenario.test_capacity_low, scenario.test_capacity_high]
        # symp_prop_values = [.5, 1]
        symp_prop_values = [.25, .5, .75]
        scenario_names = ['Low_prev', 'High_prev']
        situation_dict = {'Low_prev': {'onward': onward_transmission_vector_low,
                                       'pop': population_low,
                                       'pre_prob': test_prob_low},
                          'High_prev': {'onward': onward_transmission_vector_high,
                                       'pop': population_high,
                                       'pre_prob': test_prob_high}
                          }
        priority_allocation_options = scenario.priority_order
        tat_function_list = ['quadratic', 'linear', 'exponential']
        priority_value = priority_values[0]
        priority_order = priority_allocation_options[0]
        for capacity_value in capacity_values:
            for scenario in scenario_names:
                fig, axs = plt.subplots(1, 3, figsize=(14,8))
                scenario_plot_name = None
                if scenario == 'Low_prev':
                    scenario_plot_name = 'Outbreak response'
                if scenario == 'High_prev':
                    scenario_plot_name = 'Community transmission'
                counter = itertools.count()
                for tat_function in tat_function_list:
                    c_dict = situation_dict[scenario]
                    test_optim = TestOptimisation(priority_queue=priority_value,
                                                  onward_transmission=c_dict['onward'],
                                                  population=c_dict['pop'],
                                                  pre_test_probability=c_dict['pre_prob'],
                                                  routine_capacity=capacity_value,
                                                  symptomatic_testing_proportion=.5,
                                                  test_prioritsation_by_indication=priority_order,
                                                  tat_function=tat_function)

                    test_array, transmission_array, positive_array = \
                        test_optim.generate_onward_transmission_with_tests(max_tests_proportion=1000/capacity_value)
                    test_nparray = np.array(test_array)/100

                    c = next(counter)
                    axs[c].set_title(f'{scenario_plot_name} scenario\ntest capacity = {capacity_value/100}, TAT function: {tat_function}')
                    color = 'tab:blue'
                    color = [0,0,.6]
                    axs[c].plot(test_nparray, 100*transmission_array/max(transmission_array), color=color)
                    if c == 0:
                        axs[c].set_ylabel('Percentage of onwards transmission', color=color)
                    axs[c].set_xlabel('Number of tests per 1000')

                    # min_transmission_tests = test_nparray[np.where(transmission_array==min(transmission_array))]
                    # ax1.plot([min_transmission_tests]*2, [80,100],'k--')

                    # ax2 = ax1.twinx()
                    # color = [.25, .4, 0]
                    # ax2.plot([i/100 for i in test_array], 100*positive_array*np.array(test_array)/cases_high, color=color)
                    # ax2.set_ylabel('Percentage of infections identified', color=color)

                    # fig.tight_layout()
                    # plt.xlabel('Number of tests per 1000')
                    # plt.ylabel('Percentage reduction through contact tracing')
                    # plt.legend(['Routine capacity 2 per 1000', 'Routine capacity 4 per 1000'])
                    # axs[axs_current].set
                    # axs[axs_current].plot([capacity_value/100]*2, [0, 45],'--r')
                plt.savefig(f'{base_figure_directory}/Supplement_figures/tat_function/Cases_identified_transmission_{scenario}_capacity{capacity_value}_tatfun_{tat_function}.png')
                    # plt.show()



    if test_figure_area:
        total_population_size = 100000
        percentage_pop_by_indication_cc_sympt = (0.1, 1)
        # population = (1000, 10000, 10000)
        population = (total_population_size*percentage_pop_by_indication_cc_sympt[0]/100,
                      total_population_size*percentage_pop_by_indication_cc_sympt[1]/100,
                      total_population_size*(100 - sum(percentage_pop_by_indication_cc_sympt))/100)
        pre_test_probability = (.3, .03, .003)
        onward_transmission = (2, 3, 1, .3)
        routine_capacity = 400
        priority_capacity_proportion = .0
        priority_queue = True
        routine_tat = 10
        tat_at_fifty_percent_surge = 20
        swab_delay = 1
        symptomatic_testing_proportion = 1.
        test_prioritsation_by_indication = None


        test_optim = TestOptimisation(population=population,
                                      pre_test_probability=pre_test_probability,
                                      onward_transmission=onward_transmission,
                                      routine_capacity=routine_capacity,
                                      priority_capacity_proportion=priority_capacity_proportion,
                                      priority_queue=priority_queue,
                                      routine_tat=routine_tat,
                                      tat_at_fifty_percent_surge=tat_at_fifty_percent_surge,
                                      swab_delay=swab_delay,
                                      symptomatic_testing_proportion=symptomatic_testing_proportion,
                                      test_prioritsation_by_indication=test_prioritsation_by_indication)

        # test_optim.plot_turn_around_time()
        test_optim.plot_transmission_with_testing()

    if tat_figure:
        tat_list = [[1, 2],
                    [2, 4]]
        test_optim_1 = TestOptimisation(routine_tat=tat_list[0][0],
                                      tat_at_fifty_percent_surge=tat_list[0][1],
                                      routine_capacity=100)
        test_optim_2 = TestOptimisation(routine_tat=tat_list[1][0],
                                      tat_at_fifty_percent_surge=tat_list[1][1],
                                      routine_capacity=100)


        plt.figure(figsize=(5, 4), dpi=400)
        test_optim_1.plot_turn_around_time()
        test_optim_2.plot_turn_around_time()

        plt.plot([100, 100], [0.3, 7.5], 'r--')

        plt.legend([f'Routine TAT = {rtat}, TAT at 50% surge = {stat}'
                    for rtat, stat in tat_list] +
                   ['Routine capacity = 100'])
        plt.ylim([0, 10])
        plt.xlim([0, 200])
        plt.savefig('MS_figures/TAT_figure.png')
        plt.show()
        plt.close()

    if kretzschmar_figure:
        plt.figure(figsize=(5, 4), dpi=400)
        test_optim = TestOptimisation(swab_delay=0)
        test_optim.plot_delay_effect_on_transmission(7)
        test_optim = TestOptimisation(swab_delay=1)
        test_optim.plot_delay_effect_on_transmission(7)
        test_optim = TestOptimisation(swab_delay=2)
        test_optim.plot_delay_effect_on_transmission(7)
        test_optim = TestOptimisation(swab_delay=3)
        test_optim.plot_delay_effect_on_transmission(7)
        plt.legend([f'Swab delay = {i}' for i in range(4)])
        plt.xlabel('Turn around time (TAT)')
        plt.savefig('MS_figures/kretzschmar_results.png')
        plt.show()
        plt.close()
    pass