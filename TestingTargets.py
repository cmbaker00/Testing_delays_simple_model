import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom


def plot_pr_detect(prevalance_per_100k=1,
                   days_of_no_transmission_threshold=28,
                   target_prob=0.8,
                   max_tests=16,
                   r0=None):

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
    plt.xlabel('Tests per 1000 per day')
    plt.ylabel('Probability of detecting transmission')
    plt.ylim([0, 1])
    if type(prevalance_per_100k) is list:
        plt.title(f"Probability of detecting transmission "
                  f"with\n initial prevalence {prevalance_per_100k[0]} and\n"
                  f"Reff={r0}"
                  f" per 100k within {days_of_no_transmission_threshold} days")
        plt.savefig(f"Prob_detect_figures/detect_{prevalance_per_100k[0]}"
                    f"_r0_{r0}"
                    f"_per_100k_in{days_of_no_transmission_threshold}_days.png")
        plt.close()
    else:
        plt.title(f"Probability of detecting transmission "
                  f"with\nprevalence {prevalance_per_100k}"
                  f" per 100k within {days_of_no_transmission_threshold} days")
        plt.savefig(f"Prob_detect_figures/detect_{prevalance_per_100k}"
                    f"_per_100k_in{days_of_no_transmission_threshold}_days.png")
    # plt.show()
        plt.close()

def plot_pr_detect_increasing(prevalance_per_100k=1,
                              days_of_no_transmission_threshold=28,
                              target_prob=0.8,
                              max_tests=16,
                              r0=1,
                              serial_interval=5):
    if r0 == 1:
        plot_pr_detect(prevalance_per_100k=prevalance_per_100k,
                       days_of_no_transmission_threshold=days_of_no_transmission_threshold,
                       target_prob=target_prob,
                       max_tests=max_tests,
                       r0=None)
    else:
        daily_multiplier = r0**(1/serial_interval)
        prev_list = [prevalance_per_100k*(daily_multiplier**day) for day in range(days_of_no_transmission_threshold)]
        plot_pr_detect(prevalance_per_100k=prev_list,
                       days_of_no_transmission_threshold=days_of_no_transmission_threshold,
                       target_prob=target_prob,
                       max_tests=max_tests,
                   r0=r0)

prev_list = [0.5, 1, 2]
days_list = [14, 28]
target_prob = 0.8
max_tests = 10.
reff_list = [1, 1.1, 1.5]
for prev in prev_list:
    for day in days_list:
        for r0 in reff_list:
            print(f'running prev{prev}, days{day}, r0{r0}')
            plot_pr_detect_increasing(prevalance_per_100k=prev,
                           days_of_no_transmission_threshold=day,
                           target_prob=0.8,
                           max_tests=16,
                           r0=r0)