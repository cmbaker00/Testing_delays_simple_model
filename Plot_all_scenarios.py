from SimpleModelsModule import TestOptimisation
import matplotlib.pyplot as plt
import numpy as np
import Scenario_parameter_values as scenario

def make_onward_transmission_vector(close_contact, symptomatic, asymptomatic):
    return tuple((close_contact, symptomatic, asymptomatic) for i in range(4))


def make_population_tuple(num_close, num_symp, total_pop,
                          presenting_proporition, probability_by_indication):
    num_asymptomatic = total_pop - num_close - num_symp
    expected_cases = np.sum(np.array(probability_by_indication)*
                            np.array([num_close, num_symp, num_asymptomatic]))
    expected_cases = np.round(expected_cases, 2)
    return (num_close, num_symp*presenting_proporition, num_asymptomatic), expected_cases


def run_analysis_save_plot(priority, onward_transmission, pop, pre_prob, cap, prop_symp, scenario_name, priority_ordering=None):
    test_optim = TestOptimisation(priority_queue=priority, onward_transmission=onward_transmission,
                                  population=pop,
                                  pre_test_probability=pre_prob,
                                  routine_capacity=cap,
                                  symptomatic_testing_proportion=prop_symp,
                                  test_prioritsation_by_indication=priority_ordering)
    max_prop_plot = 3/(cap/400)
    ax, onward, pos, exp_case, num_test\
        = test_optim.make_plot_transmission_perc_post(max_test_proportion=max_prop_plot)

    rc = test_optim.routine_capacity / 100
    ax.plot([rc, rc], [50, 100], '--r')
    ax.text(rc * 1.04, 85, 'Routine capacity', rotation=270)
    priority_string = '' if priority else '_no_priority'
    priority_order_string = '' if priority_ordering == None else '_symptomatic_priority'
    plt.savefig(f'Onward_transmission_and_postivity_basic_figures/{scenario_name}_test_prop_{prop_symp}_cap_{int(cap/100)}'
                f'{priority_string}{priority_order_string}.png')
    plt.close()
    def fill_box(xmin, xmax, col=(0, 0, 0), ymin=50., ymax=100.):
        plt.fill([xmin, xmax, xmax, xmin, xmin],
                 [ymin, ymin, ymax, ymax, ymin],
                 alpha=.3,
                 color=col,
                 lw=0)
    if priority_ordering:
        if priority_ordering == (2, 1, 3):
            effective_pop = [prop_symp*i/100 for i in pop]
            effective_pop_order = [effective_pop[i - 1] for i in priority_ordering]
            test_bounds = np.cumsum([0] + effective_pop_order)
            test_bounds[-1] = 12
            col_array = [[0.2]*3, [.4]*3, [.6]*3]
            for i, col in zip(range(3), col_array):
                fill_box(test_bounds[i], test_bounds[i+1],
                         col=col)
            plt.plot(num_test, onward)

            rc = test_optim.routine_capacity / 100
            plt.plot([rc, rc], [50, 100], '--r')
            plt.text(rc * 1.04, 55, 'Routine capacity', rotation=270)

            plt.xlabel('Tests per 1000 people')
            plt.ylabel('Percentage of onwards transmission')
            plt.savefig(f'Onward_transmission_and_postivity_basic_figures/{scenario_name}_test_prop_{prop_symp}_cap_{int(cap / 100)}'
                        f'{priority_string}{priority_order_string}_onward_only.png')
            plt.show()
            plt.close()

            plt.figure()
            pos = pos*100
            fig_top = max(pos)*1.1
            for i, col in zip(range(3), col_array):
                fill_box(test_bounds[i], test_bounds[i+1],
                         ymin=0, ymax=fig_top,
                         col=col)
            plt.plot(num_test, pos)

            rc = test_optim.routine_capacity / 100
            plt.plot([rc, rc], [0, fig_top], '--r')
            plt.text(rc * 1.04, .05*fig_top, 'Routine capacity', rotation=270)

            plt.xlabel('Tests per 1000 people')
            plt.ylabel('Percentage of positive tests')
            plt.savefig(f'Onward_transmission_and_postivity_basic_figures/{scenario_name}_test_prop_{prop_symp}_cap_{int(cap / 100)}'
                        f'{priority_string}{priority_order_string}_positivity_only.png')

            plt.show()
            plt.close()
            # print(1)

if __name__ == '__main__':

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

    print(f'Daily infections = {cases_high}')

    # Low prevelance
    onward_transmission_vector_low = \
        make_onward_transmission_vector(*scenario.onward_transmission_low)

    test_prob_low = scenario.test_prob_low

    population_low, cases_low = \
        make_population_tuple(num_close=scenario.pop_low[0],
                              num_symp=scenario.pop_low[1],
                              total_pop=total_population,
                              presenting_proporition=1,
                              probability_by_indication=test_prob_low)

    print(f'Daily infections = {cases_low}')

    priority_values = [True, False]
    capacity_values = [scenario.test_capacity_low, scenario.test_capacity_high]
    symp_prop_values = [.2, .4, .5, .6, .8, 1]
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
                        run_analysis_save_plot(priority=priority_value,
                                               onward_transmission=c_dict['onward'],
                                               pop=c_dict['pop'],
                                               pre_prob=c_dict['pre_prob'],
                                               cap=capacity_value,
                                               prop_symp=symp_prop_value,
                                               scenario_name=scenario,
                                               priority_ordering=priority_order)
