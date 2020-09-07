from SimpleModelsModule import TestOptimisation
import matplotlib.pyplot as plt
import numpy as np

test_optim = TestOptimisation(priority_queue=True, onward_transmission=(2, 3, 2, 2),
                              pre_test_probability=(0.3, .005, 0.0005),
                              population=(100,600,99300),
                              routine_capacity=400,
                              swab_delay=1)

plot_infection_vs_tat = False
plot_tat_vs_tests = False
plot_inf_vs_swab_delay = False

if plot_infection_vs_tat:
    # initial onward transmission plot
    test_optim.plot_delay_effect_on_transmission(max_delay=8)
    plt.savefig('Model_explanation_figures/Onward_infection_with_tat.png')
    plt.show()

if plot_tat_vs_tests:
    test_number_array = [i+1 for i in range(1000)]
    tat = []
    for test_number in test_number_array:
        tat.append(test_optim.turn_around_time(test_number, False))

    test_number_array = np.array(test_number_array)/100

    plt.figure(num=1, figsize=(6,4))
    plt.plot(test_number_array, tat)

    plt.plot([4, 4],[0, 10],'--r')
    plt.text(4.1, 2.5, 'Routine capacity',
             rotation=-90)
    plt.xlabel('Number of tests per thousand people')
    plt.ylabel('Turn around time (days)')
    plt.savefig('Model_explanation_figures/TAT_vs_tests_per_thousand.png')
    plt.show()

if plot_inf_vs_swab_delay:
    delay = np.inf
    swab_delay_array = np.linspace(0,10,1000)
    transmission_list = []
    for swab_delay in swab_delay_array:
        t = test_optim.test_delay_effect_on_percent_future_infections(
            delay,
            swab_delay=swab_delay)
        transmission_list.append(t)
    transmission_array = np.array(transmission_list)
    plt.plot(swab_delay_array, 1 - transmission_array)
    plt.ylabel('Percentage of onwards transmission')
    plt.xlabel('Time from symptom onset to test (days)')
    plt.savefig('Model_explanation_figures/Onward_transmission_vs_swab.png')
    plt.show()