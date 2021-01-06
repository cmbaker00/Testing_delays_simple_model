import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SimpleModelsModule import TestOptimisation
import Param_values_MS as scenario
import Plot_all_scenarios

if __name__ == "__main__":
    test_figure_area = False
    tat_figure = False
    kretzschmar_figure = False
    main_figures = True

    if main_figures:
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

        priority_values = [True, False]
        capacity_values = [scenario.test_capacity_low, scenario.test_capacity_high]
        symp_prop_values = [.5, 1]
        scenario_names = ['Low_prev', 'High_prev']
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
                            Plot_all_scenarios.run_analysis_save_plot(priority=priority_value,
                                                                      onward_transmission=c_dict['onward'],
                                                                      pop=c_dict['pop'],
                                                                      pre_prob=c_dict['pre_prob'],
                                                                      cap=capacity_value,
                                                                      prop_symp=symp_prop_value,
                                                                      scenario_name=scenario,
                                                                      priority_ordering=priority_order,
                                                                      directory_name='MS_figures')



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
        tat_list = [[1, 5],
                    [4, 6]]
        test_optim_1 = TestOptimisation(routine_tat=tat_list[0][0],
                                      tat_at_fifty_percent_surge=tat_list[0][1],
                                      routine_capacity=100)
        test_optim_2 = TestOptimisation(routine_tat=tat_list[1][0],
                                      tat_at_fifty_percent_surge=tat_list[1][1],
                                      routine_capacity=100)


        plt.figure(figsize=(5, 4), dpi=400)
        test_optim_1.plot_turn_around_time()
        test_optim_2.plot_turn_around_time()

        plt.plot([100, 100], [0.3, 12], 'r--')

        plt.legend([f'Routine TAT = {rtat}, TAT at 50% surge = {stat}'
                    for rtat, stat in tat_list] +
                   ['Routine capacity = 100'])
        plt.ylim([0, 16])
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