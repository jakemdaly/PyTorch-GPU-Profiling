import csv
import pandas as pd
import numpy as np
from statistics import mean
from statistics import variance as var
import plotly.graph_objects as go
import pdb
# from utils import data_to_frame

def data_to_frame(head, BatchSize1000=False):

	frames = []

	dict_list = []

	for file_ind in range(1,51):

		frames.append(pd.read_csv(head + "%s.csv"%(file_ind), header=4))
		dur_units = frames[file_ind-1].iloc[0].to_numpy()[1]
		mem_units = frames[file_ind-1].iloc[0].to_numpy()[11]
		thp_units = frames[file_ind-1].iloc[0].to_numpy()[12]
		values_arr = frames[file_ind-1].iloc[1:].to_numpy()
		if dur_units == 'ms':
			 values_arr[:,1] = values_arr[:,1].astype(float)*1000
		if thp_units == 'GB/s':
			values_arr[:,12] = values_arr[:,12].astype(float)*1000
		if mem_units == 'MB':
			values_arr[:,11] = values_arr[:,11].astype(float)*1000
		for i in range(np.shape(values_arr)[0]):
			values = np.append(values_arr[i,:], file_ind)
			values = np.append(values, i)
			dict_list.append(dict(zip(list(frames[file_ind-1].columns) + ['Trial'] + ['CudaCode'], values)))

	cleaned_frame = pd.DataFrame(data=dict_list,columns=list(frames[0].columns) + ['Trial'] + ['CudaCode'])
	cleaned_frame.Duration = pd.to_numeric(cleaned_frame.Duration)
	cleaned_frame.Throughput = pd.to_numeric(cleaned_frame.Throughput)
	cleaned_frame.Size = pd.to_numeric(cleaned_frame.Size)

	if BatchSize1000:
		cleaned_frame.Duration = cleaned_frame.Duration/(1000)

	return cleaned_frame

# Used for plotting the kernels that appear in all layers of both BatchSize=1 and BatchSize=1000
def plot_hist(kernel_name_1, bin_size_1, kernel_name_1000, bin_size_1000, layer_dfs):
		
	fig1 = go.Figure()
	fig1000 = go.Figure()

	# BatchSize=1
	fig1.add_trace(go.Histogram(x=layer_dfs['l1_df'].query('Name==\'%s\''%kernel_name_1).Duration, name='L1: '+kernel_name_1[:15], xbins=dict(size=bin_size_1), marker_color='lightblue'))
	fig1.add_trace(go.Histogram(x=layer_dfs['l2_df'].query('Name==\'%s\''%kernel_name_1).Duration, name='L2: '+kernel_name_1[:15], xbins=dict(size=bin_size_1), marker_color='blue'))
	fig1.add_trace(go.Histogram(x=layer_dfs['l3_df'].query('Name==\'%s\''%kernel_name_1).Duration, name='L3: '+kernel_name_1[:15], xbins=dict(size=bin_size_1), marker_color='darkblue'))
	
	# BatchSize=1000
	fig1000.add_trace(go.Histogram(x=layer_dfs['l1_df_1000'].query('Name==\'%s\''%kernel_name_1000).Duration, name='L1: '+kernel_name_1000[:15], xbins=dict(size=bin_size_1000), marker_color='lightgreen'))
	fig1000.add_trace(go.Histogram(x=layer_dfs['l2_df_1000'].query('Name==\'%s\''%kernel_name_1000).Duration, name='L2: '+kernel_name_1000[:15], xbins=dict(size=bin_size_1000), marker_color='green'))
	fig1000.add_trace(go.Histogram(x=layer_dfs['l3_df_1000'].query('Name==\'%s\''%kernel_name_1000).Duration, name='L3: '+kernel_name_1000[:15], xbins=dict(size=bin_size_1000), marker_color='darkgreen'))
	
	# Additional formatting and titles
	fig1.update_layout(barmode='overlay',
					   title_text='BatchSize=1 Kernel Runtime Variations',
					   xaxis_title_text='Kernel Duration (us)',
					   yaxis_title_text='Number of Occurances (out of 50 trials)'
					  )
	fig1.update_traces(opacity=.75)
	
	fig1000.update_layout(barmode='overlay',
					   title_text='BatchSize=1 Kernel Runtime Variations',
					   xaxis_title_text='Kernel Duration (us)',
					   yaxis_title_text='Number of Occurances (out of 50 trials)'
					  )
	fig1000.update_traces(opacity=.75)

	fig1.show()
	fig1000.show()

# Used for plotting the main workload
def plot_hist_workload_kernels(layer_dfs):
	fig1 = go.Figure()
	fig1000 = go.Figure()

	###### BatchSize=1, Layer1
	fig1.add_trace(go.Histogram(x=layer_dfs['l1_df'].query('Name==\'%s\''%layer_dfs['names_l1_1'][1]).Duration, name='L1: '+layer_dfs['names_l1_1'][1][:15], marker_color='lightblue'))
	fig1.add_trace(go.Histogram(x=layer_dfs['l1_df'].query('Name==\'%s\''%layer_dfs['names_l1_1'][2]).Duration, name='L1: '+layer_dfs['names_l1_1'][2][:15], marker_color='blue'))
	
	# BatchSize=1, Layer2
	fig1.add_trace(go.Histogram(x=layer_dfs['l2_df'].query('Name==\'%s\''%layer_dfs['names_l2_1'][1]).Duration, name='L2: '+layer_dfs['names_l2_1'][1][:15], marker_color='lightgreen'))
	fig1.add_trace(go.Histogram(x=layer_dfs['l2_df'].query('Name==\'%s\''%layer_dfs['names_l2_1'][2]).Duration, name='L2: '+layer_dfs['names_l2_1'][2][:15], marker_color='green'))
	
	# BatchSize=1, Layer3
	fig1.add_trace(go.Histogram(x=layer_dfs['l3_df'].query('Name==\'%s\''%layer_dfs['names_l3_1'][1]).Duration, name='L3: '+layer_dfs['names_l3_1'][1][:15], marker_color='lightsalmon'))
	fig1.add_trace(go.Histogram(x=layer_dfs['l3_df'].query('Name==\'%s\''%layer_dfs['names_l3_1'][2]).Duration, name='L3: '+layer_dfs['names_l3_1'][2][:15], marker_color='red'))
	
	
	###### BatchSize=1000, Layer1
	fig1000.add_trace(go.Histogram(x=layer_dfs['l1_df_1000'].query('Name==\'%s\''%layer_dfs['names_l1_1000'][1]).Duration, name='L1: '+layer_dfs['names_l1_1000'][1][:15], marker_color='lightblue'))
	fig1000.add_trace(go.Histogram(x=layer_dfs['l1_df_1000'].query('Name==\'%s\''%layer_dfs['names_l1_1000'][2]).Duration, name='L1: '+layer_dfs['names_l1_1000'][2][:15], marker_color='blue'))
	
	# BatchSize=1, Layer2
	fig1000.add_trace(go.Histogram(x=layer_dfs['l2_df_1000'].query('Name==\'%s\''%layer_dfs['names_l2_1000'][1]).Duration, name='L2: '+layer_dfs['names_l2_1000'][1][:15], marker_color='lightgreen'))
	fig1000.add_trace(go.Histogram(x=layer_dfs['l2_df_1000'].query('Name==\'%s\''%layer_dfs['names_l2_1000'][2]).Duration, name='L2: '+layer_dfs['names_l2_1000'][2][:15], marker_color='green'))
	fig1000.add_trace(go.Histogram(x=layer_dfs['l2_df_1000'].query('Name==\'%s\''%layer_dfs['names_l2_1000'][3]).Duration, name='L2: '+layer_dfs['names_l2_1000'][3][:15], marker_color='darkgreen'))
	fig1000.add_trace(go.Histogram(x=layer_dfs['l2_df_1000'].query('Name==\'%s\''%layer_dfs['names_l2_1000'][4]).Duration, name='L2: '+layer_dfs['names_l2_1000'][4][:15], marker_color='forestgreen'))
	
	# BatchSize=1, Layer3
	fig1000.add_trace(go.Histogram(x=layer_dfs['l3_df_1000'].query('Name==\'%s\''%layer_dfs['names_l3_1000'][1]).Duration, name='L3: '+layer_dfs['names_l3_1000'][1][:15], marker_color='lightsalmon'))
	fig1000.add_trace(go.Histogram(x=layer_dfs['l3_df_1000'].query('Name==\'%s\''%layer_dfs['names_l3_1000'][2]).Duration, name='L3: '+layer_dfs['names_l3_1000'][2][:15], marker_color='red'))
	
	
	# Additional formatting and titles
	fig1.update_layout(barmode='overlay',
					   title_text='Kernel Runtime Variations (BatchSize=1, main compute kernels)',
					   xaxis_title_text='Kernel Duration (us)',
					   yaxis_title_text='Number of Occurances (out of 50 trials)'
					  )
	fig1.update_traces(opacity=.75)
	
	fig1000.update_layout(barmode='overlay',
					   title_text='Kernel Runtime Variations (BatchSize=1000; main compute kernels)',
					   xaxis_title_text='Kernel Duration (us)',
					   yaxis_title_text='Number of Occurances (out of 50 trials)'
					  )
	fig1000.update_traces(opacity=.75)

	fig1.show()
	fig1000.show()

# Use for extracting the throughputs for load weights & data, and dumping output (all off chip transfers)
def calc_throughput(dataframe, weight_cudacall_id, input_cudacall_id, output_cudacall_id):
	
	thp_dict = {}
	
	thp_dict['w_size (KB)'] = mean(dataframe.query('CudaCode==%s'%weight_cudacall_id).Size)
	thp_dict['w_throughput (MB/s)'] = mean(dataframe.query('CudaCode==%s'%weight_cudacall_id).Throughput)
	
	thp_dict['i_size (KB)'] = mean(dataframe.query('CudaCode==%s'%input_cudacall_id).Size)
	thp_dict['i_throughput (MB/s)'] = mean(dataframe.query('CudaCode==%s'%input_cudacall_id).Throughput)
	
	thp_dict['o_size (KB)'] = mean(dataframe.query('CudaCode==%s'%output_cudacall_id).Size)
	thp_dict['o_throughput (MB/s)'] = mean(dataframe.query('CudaCode==%s'%output_cudacall_id).Throughput)
	
	return(thp_dict)
	
	

def get_kernels(layer_df):
	
	names = []
	names_raw = layer_df.Name.unique()
	
	for nr in names_raw:
		if not ('mem' in nr):
			names.append(nr)
	return(names)
	
def coeffvar(data_frame):
	return (var(data_frame)**(1/2))/mean(data_frame)