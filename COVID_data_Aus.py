import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test_number_hist(url, max_num, plt_title, save_name,
                     save_dir='Data_plots/', scaling=1.,
                     xlabel_extra_text=''):
    data_tests = pd.read_html(url)
    data_net = data_tests[1]['NET']
    data_net = [int(i) for i in data_net if i != '-']
    data_net = np.array(data_net)*scaling
    plt.hist(data_net, np.linspace(0, max_num, 20))

    percentile_80 = np.percentile(data_net, 80)
    print(f"{save_name} - 80th percentile: {percentile_80}")
    plt.plot([percentile_80] * 2, plt.ylim(), '--')
    plt.xlim(0, max_num)
    plt.xlabel(f'Tests per day{xlabel_extra_text}')
    plt.ylabel('Frequency')
    plt.title(plt_title)
    plt.savefig(f"{save_dir}{save_name}")
    plt.close()

vec = [
    ['vic', 45000, 'Victoria', 5000000],
       ['qld', 30000, 'Queensland', 2500000],
       ['nsw', 40000, 'New South Wales', 5300000]
       ]

for l in vec:
    test_number_hist(f'https://covidlive.com.au/report/daily-tests/{l[0]}',
                     max_num=l[1],
                     plt_title=l[2],
                     save_name=f'{l[0]}_absolute.png')
    current_scale = 1000/l[3]
    test_number_hist(f'https://covidlive.com.au/report/daily-tests/{l[0]}',
                     max_num=10,
                     plt_title=l[2],
                     save_name=f'{l[0]}_per_thousand.png',
                     scaling=current_scale,
                     xlabel_extra_text=' per thousand')