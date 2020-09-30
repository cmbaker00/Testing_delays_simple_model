import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from itertools import count


def plot_pr_detect(prevalance_per_100k=1,
                   days_of_no_transmission_threshold=28,
                   target_prob=0.8,
                   max_tests=16,
                   r0=None,
                   include_plot_labelling=True):

    tests_per_k_per_day_range = range(max_tests + 1)

    try:
        prevalance_per_100k = float(prevalance_per_100k)
        pr_detect = [1-binom.cdf(0,tests*1000,
                                 prevalance_per_100k/100000)
                     **days_of_no_transmission_threshold
                     for tests in tests_per_k_per_day_range]
    except TypeError:
        if r0 is None:
            raise ValueError("Must input R0")
        prevalance_per_100k = list(prevalance_per_100k)
        pr_detect = [1-np.prod([binom.cdf(0 ,tests*1000,
                                          current_prev/100000)
                              for current_prev
                              in prevalance_per_100k])
                     for tests in tests_per_k_per_day_range]
        days_of_no_transmission_threshold = len(prevalance_per_100k)

    plt.plot(tests_per_k_per_day_range, pr_detect, 'o')
    if target_prob:
        plt.plot([0, max_tests], [target_prob]*2, '--')
    if include_plot_labelling:
        plt.xlabel('Tests per 1000 per day')
        plt.ylabel('Probability of detecting transmission')
    plt.ylim([0, 1])

def plot_pr_detect_increasing(prevalance_per_100k=1,
                              days_of_no_transmission_threshold=28,
                              target_prob=0.8,
                              max_tests=16,
                              r0=1,
                              serial_interval=5,
                              include_plot_labelling=True):
    if r0 == 1:
        plot_pr_detect(prevalance_per_100k=prevalance_per_100k,
                       days_of_no_transmission_threshold=days_of_no_transmission_threshold,
                       target_prob=target_prob,
                       max_tests=max_tests,
                       r0=None,
                       include_plot_labelling=include_plot_labelling)
    else:
        daily_multiplier = r0**(1/serial_interval)
        prev_list = [prevalance_per_100k*(daily_multiplier**day) for day in range(days_of_no_transmission_threshold)]
        plot_pr_detect(prevalance_per_100k=prev_list,
                       days_of_no_transmission_threshold=days_of_no_transmission_threshold,
                       target_prob=target_prob,
                       max_tests=max_tests,
                       r0=r0,
                       include_plot_labelling=include_plot_labelling)

if __name__ == '__main__':
    create_single_figures = False
    create_multi_panel_figures = True

    prev_list = [0.5, 1, 2]
    days_list = [14, 28]
    target_prob = 0.8
    max_tests = 10.
    reff_list = [1, 1.1, 1.5]
    if create_multi_panel_figures:
        for r0 in reff_list:
            fig, ax = plt.subplots(len(prev_list), len(days_list))
            for prev, prev_index in zip(prev_list, count()):
                for day, day_index in zip(days_list, count()):
                    plt.axes(ax[prev_index, day_index])
                    plot_pr_detect_increasing(prevalance_per_100k=prev,
                                   days_of_no_transmission_threshold=day,
                                   target_prob=0.8,
                                   max_tests=16,
                                   r0=r0,
                                   include_plot_labelling=False)
                    if prev_index+1 == len(prev_list):
                        plt.xlabel('Tests per 1,000 per day')
                    if (prev_index) == int(len(prev_list)/2) and day_index == 0:
                        plt.ylabel('Probability of detection')
                    if prev_index == 0:
                        plt.title(f'Detection within {day} days')
                    plt.text(10, 0.1, f'{prev} per 100k')
            plt.savefig(f'Prob_detect_figures/multi_r{r0}.png')
            plt.close()




    if create_single_figures:
        for prev in prev_list:
            for day in days_list:
                for r0 in reff_list:
                    print(f'running prev{prev}, days{day}, r0{r0}')
                    plot_pr_detect_increasing(prevalance_per_100k=prev,
                                   days_of_no_transmission_threshold=day,
                                   target_prob=0.8,
                                   max_tests=16,
                                   r0=r0)

                    if r0 == 1:
                        plt.title(f"Probability of detecting transmission "
                                  f"with\nprevalence {prev}"
                                  f" per 100k within {day} days")
                        plt.savefig(f"Prob_detect_figures/detect_{prev}"
                                    f"_per_100k_in{day}_days.png")
                    else:
                        plt.title(f"Probability of detecting transmission "
                                  f"with\n initial prevalence {prev} and\n"
                                  f"Reff={r0}"
                                  f" per 100k within {day} days")
                        plt.savefig(f"Prob_detect_figures/detect_{prev}"
                                    f"_r0_{r0}"
                                    f"_per_100k_in{day}_days.png")
                    # plt.show()
                    plt.close()