import csv
import pandas as pd
import numpy as np

frames = []
data = []

dict_list_l1 = []

for file_ind in range(1,51):

	frames.append(pd.read_csv("l1\\trace_l1_20200329_%s.csv"%file_ind, header=4))
	values_arr = frames[file_ind-1].iloc[1:].to_numpy()
	for i in range(np.shape(values_arr)[0]):
		dict_list_l1.append(dict(zip(list(frames[file_ind-1].columns), values_arr[i,:])))


cleaned_frame_l1 = pd.DataFrame(data=dict_list_l1,columns=frames[0].columns)
cleaned_frame_l1.Duration = pd.to_numeric(cleaned_frame_l1.Duration)

kernel_names = cleaned_frame_l1.Name.unique()

stats_columns = ['avg_dur_l1', 'avg_dur_l2', 'avg_dur_l3']
stats_df = pd.DataFrame(columns=stats_columns)

for kernel in kernel_names:
	kernel_df = cleaned_frame_l1[cleaned_frame_l1['Name'] == kernel]
	mean_l1 = kernel_df.Duration.sum()/50
	kernel_df = cleaned_frame_l2[cleaned_frame_l2['Name'] == kernel]
	mean_l2 = kernel_df.Duration.sum()/50
	kernel_df = cleaned_frame_l3[cleaned_frame_l3['Name'] == kernel]
	mean_l3 = kernel_df.Duration.sum()/50

	stats_df.append({stats_columns[0]: mean_l1, stats_columns[1]:mean_l2, stats_columns[2]:mean_l3}, ignore_index=True)







print(list(cleaned_frame_l1.columns))
stats_columns = ['avg_dur_l1', 'avg_dur_l2', 'avg_dur_l3']
stats_df = pd.DataFrame(columns=stats_columns)

#


stats_df.append()



