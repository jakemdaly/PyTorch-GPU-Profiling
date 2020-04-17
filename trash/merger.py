import csv
import pandas as pd
import numpy as np

frames = []
data = []

dict_list_l1 = []

for file_ind in range(0,2):

	frames.append(pd.read_csv("trace_20200329.csv"))
	values_arr = frames[file_ind].iloc[5:].to_numpy()
	for i in range(np.shape(values_arr)[0]):
		dict_list_l1.append(dict(zip(list(frames[0].iloc[3]), values_arr[i,:])))


cleaned_frame = pd.DataFrame(data=dict_list_l1,columns=list(frames[0].iloc[3]))



# megaframe.to_csv(r'megaframe.csv')

	# with open(f'trace_20200329.csv') as f, open('out.csv', 'wb') as f_out:
	# 	reader = csv.reader(f)
	# 	writer = csv.writer(f_out)
	# 	reader = list(reader)
	# 	for line in range(6,63):
	# 		print(line)
	# 		writer.writerow(reader[line])