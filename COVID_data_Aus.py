import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# def test_number_hist(url, max_num, plt_title, save_name, save_dir='Data_plots/')
#


vic_tests = pd.read_html('https://covidlive.com.au/report/daily-tests/vic')
vic_data = vic_tests[1]['NET']
vic_data = [int(i) for i in vic_data if i != '-']
plt.hist(vic_data, np.arange(0, 45000, 2500))

vic_80_percentile = np.percentile(vic_data, 80)
plt.plot([vic_80_percentile]*2, plt.ylim(),'--')
plt.xlim(0, 45000)
plt.title('Victoria')
plt.show()
plt.close()


qld_tests = pd.read_html('https://covidlive.com.au/report/daily-tests/qld')
qld_data = qld_tests[1]['NET']
qld_data = [int(i) for i in qld_data if i != '-']
plt.hist(qld_data, np.arange(0, 30000, 2000))

qld_80_percentile = np.percentile(qld_data, 80)
plt.plot([qld_80_percentile]*2, plt.ylim(),'--')

plt.xlim(0, 30000)
plt.title('Queensland')
plt.show()

nsw_tests = pd.read_html('https://covidlive.com.au/report/daily-tests/nsw')
nsw_data = nsw_tests[1]['NET']
nsw_data = [int(i) for i in nsw_data if i != '-']
plt.hist(nsw_data, np.arange(0, 40000, 2000))

nsw_80_percentile = np.percentile(nsw_data, 80)
plt.plot([nsw_80_percentile]*2, plt.ylim(),'--')

plt.xlim(0, 40000)
plt.title('New South Wales')
plt.show()