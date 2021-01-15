import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('NHS_T_T_data_table_w31_table4.xlsx')
data_array = np.array(df)

date_row = data_array[1]
col_flag = []
for entry in date_row:
    if not isinstance(entry, str):
        if np.isnan(entry):
            col_flag.append(False)
    else:
        if entry[-5:] == '12/20' or entry[-5:] == '11/20':
            col_flag.append(True)
        else:
            col_flag.append(False)

data_index = np.where(col_flag)[0]

num_tests = []
perc_24 = []
perc_48 = []
perc_72 = []
perc_long = []

rows_num_tests = [13, 14, 15, 16, 22, 23]
for i in data_index:
    num_tests.append(sum([data_array[j][i] for j in rows_num_tests]))
    perc_24.append(data_array[17][i])
    perc_48.append(data_array[18][i])
    perc_72.append(data_array[19][i])
    perc_long.append(data_array[20][i])

fig, axs = plt.subplots(2, 2, figsize = (10,8))
axs[0, 0].scatter(num_tests, perc_24)
axs[0, 1].scatter(num_tests, perc_48)
axs[1, 0].scatter(num_tests, perc_72)
axs[1, 1].scatter(num_tests, perc_long)

axs[0, 0].set_xticks([150000,200000,250000,300000,350000])
axs[1, 0].set_xticks([150000,200000,250000,300000,350000])
axs[0, 1].set_xticks([150000,200000,250000,300000,350000])
axs[1, 1].set_xticks([150000,200000,250000,300000,350000])

axs[1, 1].set_xlabel('Number of tests in week')
axs[1, 0].set_xlabel('Number of tests in week')

axs[0, 0].set_title('% test results less than 24h')
axs[0, 1].set_title('% test results between 24h and 48h')
axs[1, 0].set_title('% test results between 48h and 72h')
axs[1, 1].set_title('% test results greater than 72h')

plt.savefig('MS_figures/Supplement_figures/UK_data.png')
plt.show()

print(len(data_index))