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
    fig1.add_trace(go.Histogram(x=layer_dfs['l1_df'].query('Name==\'%s\''%layer_dfs['names_l1_1'][1]).Duration, name='L1:'+layer_dfs['names_l1_1'][1][:15], marker_color='lightblue'))
    fig1.add_trace(go.Histogram(x=layer_dfs['l1_df'].query('Name==\'%s\''%layer_dfs['names_l1_1'][2]).Duration, name='L1:'+layer_dfs['names_l1_1'][2][:15], marker_color='blue'))

    # BatchSize=1, Layer2
    fig1.add_trace(go.Histogram(x=layer_dfs['l2_df'].query('Name==\'%s\''%layer_dfs['names_l2_1'][1]).Duration, name='L2:'+layer_dfs['names_l2_1'][1][:15], marker_color='lightgreen'))
    fig1.add_trace(go.Histogram(x=layer_dfs['l2_df'].query('Name==\'%s\''%layer_dfs['names_l2_1'][2]).Duration, name='L2:'+layer_dfs['names_l2_1'][2][:15], marker_color='green'))

    # BatchSize=1, Layer3
    fig1.add_trace(go.Histogram(x=layer_dfs['l3_df'].query('Name==\'%s\''%layer_dfs['names_l3_1'][1]).Duration, name='L3:'+layer_dfs['names_l3_1'][1][:15], marker_color='lightsalmon'))
    fig1.add_trace(go.Histogram(x=layer_dfs['l3_df'].query('Name==\'%s\''%layer_dfs['names_l3_1'][2]).Duration, name='L3:'+layer_dfs['names_l3_1'][2][:15], marker_color='red'))


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



def fileToFrame(fileName, layer, batchSize, trial):
    # valid and working. will override this function with LayerDimensionalityTesting
    
    frame = pd.read_csv(fileName, header=4)
    dur_units = frame.iloc[0].to_numpy()[1]
    mem_units = frame.iloc[0].to_numpy()[11]
    thp_units = frame.iloc[0].to_numpy()[12]
    values_arr = frame.iloc[1:].to_numpy()
    if dur_units == 'ms':
         values_arr[:,1] = values_arr[:,1].astype(float)*1000
    if thp_units == 'GB/s':
        values_arr[:,12] = values_arr[:,12].astype(float)*1000
    if mem_units == 'MB':
        values_arr[:,11] = values_arr[:,11].astype(float)*1000
    if mem_units == 'GB':
        values_arr[:,11] = values_arr[:,11].astype(float)*1000000
    series = []
    for i in range(np.shape(values_arr)[0]):
        values = np.append(values_arr[i,:], trial)
        values = np.append(values, i)
        values = np.append(values, batchSize)
        values = np.append(values, layer)
        series.append((dict(zip(list(frame.columns) + ['Trial'] + ['CudaCode'] + ['batchSize'] + ['Layer'], values))))

    cleaned_frame = pd.DataFrame(data=series, columns=list(frame.columns) + ['Trial'] + ['CudaCode'] + ['batchSize'] + ['Layer'])
    cleaned_frame.Duration = pd.to_numeric(cleaned_frame.Duration)
    cleaned_frame.Throughput = pd.to_numeric(cleaned_frame.Throughput)
    cleaned_frame.Size = pd.to_numeric(cleaned_frame.Size)
    cleaned_frame = cleaned_frame.rename(columns={'Name':'Kernel'})

    return(cleaned_frame)

def fileToFrameFLOPS(fileName, layer, batchSize, trial):
    # valid and working. will override this function with LayerDimensionalityTesting
    frame = pd.read_csv(fileName, header=5)

    values_arr = frame.iloc[0:].to_numpy()

    series = []
    for i in range(np.shape(values_arr)[0]):

        for j in range(5,8):
            data = values_arr[i][j]
            if 'MB/s' in data:
                values_arr[i][j] = float(values_arr[i][j][:-4])
            if 'GB/s' in data:
                values_arr[i][j] = float(values_arr[i][j][:-4])*1000
            if '%' in data:
                values_arr[i][j] = float(values_arr[i][j][:-1])/100
            values_arr[i][j] = float(values_arr[i][j])
            assert(isinstance(values_arr[i][j], float))




        values = np.append(values_arr[i,:], trial)
        values = np.append(values, i)
        values = np.append(values, batchSize)
        values = np.append(values, layer)
        series.append((dict(zip(list(frame.columns) + ['Trial'] + ['MetricCode'] + ['batchSize'] + ['Layer'], values))))

    cleaned_frame = pd.DataFrame(data=series, columns=list(frame.columns) + ['Trial'] + ['MetricCode'] + ['batchSize'] + ['Layer'])

    cols = cleaned_frame.columns
    cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, (str)) else x)
    cleaned_frame.columns = cols

    return(cleaned_frame)

def get_stats(flop_metrics_frame, trace_frame, layers):
    LayerKernelStatsDict = {}
    nl = 50
    Stats = ['achieved_occupancy', 'sm_efficiency', 'flops', 'kern_runtime_avg', 'kern_runtime_std']
    for l in layers:

        LayerKernelStatsDict['L%s'%l] = {}

        kernels = list(flop_metrics_frame.query('Layer==%s'%l)['Kernel'].unique())
        for k in kernels:
            LayerKernelStatsDict['L%s'%l]['L%s: %s'%(l,k[:nl])] = {}

        for k in kernels:

            LayerKernelStatsDict['L%s'%l]['L%s: %s'%(l,k[:nl])][Stats[0]] = list(flop_metrics_frame.query('Layer==%s and Kernel==\'%s\' and Metric_Name==\'achieved_occupancy\''%(l,k))['Avg'])[0]
            LayerKernelStatsDict['L%s'%l]['L%s: %s'%(l,k[:nl])][Stats[1]] = list(flop_metrics_frame.query('Layer==%s and Kernel==\'%s\' and Metric_Name==\'sm_efficiency\''%(l,k))['Avg'])[0]
            flops = flop_metrics_frame[flop_metrics_frame['Metric_Name'].str.contains('flop_count')]
#             pdb.set_trace()
            LayerKernelStatsDict['L%s'%l]['L%s: %s'%(l,k[:nl])][Stats[2]] = int(flops.query('Layer==%s and Kernel==\'%s\''%(l,k))['Avg'].sum())
            
            LayerKernelStatsDict['L%s'%l]['L%s: %s'%(l,k[:nl])][Stats[3]] = trace_frame.query('Layer==%s and Kernel==\'%s\''%(l,k))['Duration'].mean()
            LayerKernelStatsDict['L%s'%l]['L%s: %s'%(l,k[:nl])][Stats[4]] = trace_frame.query('Layer==%s and Kernel==\'%s\''%(l,k))['Duration'].std()

            LayerKernelStatsDict['L%s'%l]['L%s: %s'%(l,k[:nl])]['Layer'] = l

    # Transform dictionary into the two dataframes we care about:

    stats = []
    for l in layers:
        stats.append(pd.DataFrame.from_dict(LayerKernelStatsDict['L%s'%l],orient='index'))
    kernel_stats_frame = pd.concat(stats)
#     pdb.set_trace()
    layer_stats_frame = kernel_stats_frame.groupby('Layer').sum()

    # Can't just sum up achieved_occupancy and sm_efficiency, so we need to compute weighted average:
    total_runtimes = dict(layer_stats_frame.kern_runtime_avg)
    total_runtimes_ = [0]*10 # Ten is arbitrary. Value must be large enough to store a value for every potential layer of a network (eg. MNIST -> 3, but PGAN requires -> 6). 10 is to be safe
    for layer in total_runtimes:
        total_runtimes_[layer-1] = total_runtimes[layer]

    #######

    new_sm_eff = [0]*10 # see comment above
    new_ach_occ = [0]*10
    for row in range(len(kernel_stats_frame)):
#         pdb.set_trace()
        new_sm_eff[int(kernel_stats_frame.iloc[row]['Layer'])-1] += kernel_stats_frame.iloc[row]['sm_efficiency']*kernel_stats_frame.iloc[row]['kern_runtime_avg']/total_runtimes_[int(kernel_stats_frame.iloc[row]['Layer'])-1]
        new_ach_occ[int(kernel_stats_frame.iloc[row]['Layer'])-1] += kernel_stats_frame.iloc[row]['achieved_occupancy']*kernel_stats_frame.iloc[row]['kern_runtime_avg']/total_runtimes_[int(kernel_stats_frame.iloc[row]['Layer'])-1]
        
    ######
    
    new_sm_eff = [i for i in new_sm_eff if i != 0]
    new_ach_occ = [i for i in new_ach_occ if i != 0]
#     pdb.set_trace()
    layer_stats_frame['achieved_occupancy']=new_ach_occ
    layer_stats_frame['sm_efficiency']=new_sm_eff
    layer_stats_frame['bytes_fetched'] = list(trace_frame.query('(Kernel==\'[CUDA memcpy HtoD]\' or Kernel==\'[CUDA memcpy DtoH]\') and Trial==1').groupby('Layer').sum().Size)
    layer_stats_frame['throughput'] = list((layer_stats_frame["flops"]/1e9)/(layer_stats_frame["kern_runtime_avg"]/1e6))
    layer_stats_frame['arithmetic_intensity'] = list((layer_stats_frame["flops"])/(layer_stats_frame["bytes_fetched"]*1e3))
    layer_stats_frame = layer_stats_frame.rename(columns={'kern_runtime_avg':'layer_runtime_avg', 'kern_runtime_std':'layer_runtime_std'})
    return layer_stats_frame, kernel_stats_frame, LayerKernelStatsDict


def fileToFrameLayerDimProf(fileName, layer, batchSize, trial, LayerDepthIn, LayerDepthOut, K, dimToBeVaried):
    #dimToBeVaried = "IN", "OUT", "K"
    
    if not (dimToBeVaried=="IN" or dimToBeVaried=="OUT" or dimToBeVaried=="K"):
        print("Bad value for dimToBeVaried")
        assert(False)
    frame = pd.read_csv(fileName, header=4)
    dur_units = frame.iloc[0].to_numpy()[1]
    mem_units = frame.iloc[0].to_numpy()[11]
    thp_units = frame.iloc[0].to_numpy()[12]
    values_arr = frame.iloc[1:].to_numpy()
    if dur_units == 'ms':
        values_arr[:,1] = values_arr[:,1].astype(float)*1000
    if dur_units == 's':
        values_arr[:,1] = values_arr[:,1].astype(float)*1000000
        assert(False) # I'd like to know about this 
    if thp_units == 'KB/s':
        values_arr[:,12] = values_arr[:,12].astype(float)/1000
        assert(False) # I'd like to know about this 
    if thp_units == 'GB/s':
        values_arr[:,12] = values_arr[:,12].astype(float)*1000
    if mem_units == 'B':
        values_arr[:,11] = values_arr[:,11].astype(float)/1000
    if mem_units == 'MB':
        values_arr[:,11] = values_arr[:,11].astype(float)*1000
    if mem_units == 'GB':
        values_arr[:,11] = values_arr[:,11].astype(float)*1000000
    series = []
    for i in range(np.shape(values_arr)[0]):
        values = np.append(values_arr[i,:], trial)
        values = np.append(values, i)
        values = np.append(values, batchSize)
        values = np.append(values, layer)
        values = np.append(values, LayerDepthIn)
        values = np.append(values, LayerDepthOut)
        values = np.append(values, K)
        values = np.append(values, dimToBeVaried)
        series.append((dict(zip(list(frame.columns) + ['Trial'] + ['CudaCode'] + ['batchSize'] + ['Layer'] + ['LayerDepthIn'] + ['LayerDepthOut'] + ['K'] + ['VariedDimension'], values))))

    cleaned_frame = pd.DataFrame(data=series, columns=list(frame.columns) + ['Trial'] + ['CudaCode'] + ['batchSize'] + ['Layer'] + ['LayerDepthIn'] + ['LayerDepthOut'] + ['K'] + ['VariedDimension'])
    cleaned_frame.Duration = pd.to_numeric(cleaned_frame.Duration)
    cleaned_frame.Throughput = pd.to_numeric(cleaned_frame.Throughput)
    cleaned_frame.Size = pd.to_numeric(cleaned_frame.Size)
    cleaned_frame = cleaned_frame.rename(columns={'Name':'Kernel'})

    return(cleaned_frame)

def fileToFrameLayerDimProfFLOPS(fileName, layer, batchSize, LayerDepthIn, LayerDepthOut, K, dimToBeVaried):
    # valid and working. will override this function with LayerDimensionalityTesting
    #dimToBeVaried = "IN", "OUT", "K"
    
    if not (dimToBeVaried=="IN" or dimToBeVaried=="OUT" or dimToBeVaried=="K"):
        print("Bad value for dimToBeVaried")
        assert(False)
    
    
    frame = pd.read_csv(fileName, header=5)

    values_arr = frame.iloc[0:].to_numpy()

    series = []
    for i in range(np.shape(values_arr)[0]):

        for j in range(5,8):
            data = values_arr[i][j]
            if 'MB/s' in data:
                values_arr[i][j] = float(values_arr[i][j][:-4])
            if 'GB/s' in data:
                values_arr[i][j] = float(values_arr[i][j][:-4])*1000
            if '%' in data:
                values_arr[i][j] = float(values_arr[i][j][:-1])/100
            values_arr[i][j] = float(values_arr[i][j])
            assert(isinstance(values_arr[i][j], float))




#         values = np.append(values_arr[i,:], trial)
        values = np.append(values_arr[i,:], i)
        values = np.append(values, batchSize)
        values = np.append(values, layer)
        values = np.append(values, LayerDepthIn)
        values = np.append(values, LayerDepthOut)
        values = np.append(values, K)
        values = np.append(values, dimToBeVaried)
        series.append((dict(zip(list(frame.columns) + ['MetricCode'] + ['batchSize'] + ['Layer'] + ['LayerDepthIn'] + ['LayerDepthOut'] + ['K'] + ['VariedDimension'], values))))

    cleaned_frame = pd.DataFrame(data=series, columns=list(frame.columns) + ['MetricCode'] + ['batchSize'] + ['Layer'] + ['LayerDepthIn'] + ['LayerDepthOut'] + ['K'] + ['VariedDimension'])

    cols = cleaned_frame.columns
    cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, (str)) else x)
    cleaned_frame.columns = cols

    return(cleaned_frame)


def get_stats_varyingInChanDepth(flop_metrics_frame, trace_frame, layers, din):
    
    # din = list of values that input was varied over
    layerstats_ldp = []
    kernelstats_ldp = []
    
    for d in din:
        
        flop_metrics_frame_tailored = flop_metrics_frame.query('LayerDepthIn==%s and VariedDimension==\"IN\"'%d)
        trace_frame_tailored = trace_frame.query('LayerDepthIn==%s and VariedDimension==\"IN\"'%d)
#         pdb.set_trace()
        layer_stats, kern_stats, dictionary = get_stats(flop_metrics_frame_tailored, trace_frame_tailored, layers)
        layer_stats['LayerDepthIn'] = [d]*len(layer_stats)
        kern_stats['LayerDepthIn'] = [d]*len(kern_stats)
        
        layerstats_ldp.append(layer_stats)
        kernelstats_ldp.append(kern_stats)
    
    layerstats_df = pd.concat(layerstats_ldp)
    kernelstats_df = pd.concat(kernelstats_ldp)
    
    return layerstats_df, kernelstats_df

def get_stats_varyingOutChanDepth(flop_metrics_frame, trace_frame, layers, dout):
    
    # dout = list of values that input was varied over
    
    layerstats_ldp = []
    kernelstats_ldp = []
    
    for d in dout:
        
        flop_metrics_frame_tailored = flop_metrics_frame.query('LayerDepthOut==%s and VariedDimension==\"OUT\"'%d)
        trace_frame_tailored = trace_frame.query('LayerDepthOut==%s and VariedDimension==\"OUT\"'%d)
#         pdb.set_trace()
        layer_stats, kern_stats, dictionary = get_stats(flop_metrics_frame_tailored, trace_frame_tailored, layers)
        layer_stats['LayerDepthOut'] = [d]*len(layer_stats)
        kern_stats['LayerDepthOut'] = [d]*len(kern_stats)
        
        layerstats_ldp.append(layer_stats)
        kernelstats_ldp.append(kern_stats)
    
    layerstats_df = pd.concat(layerstats_ldp)
    kernelstats_df = pd.concat(kernelstats_ldp)
    
    return layerstats_df, kernelstats_df

def get_stats_varyingK(flop_metrics_frame, trace_frame, layers, dk):
    
    # dk = list of values that input was varied over
    
    layerstats_ldp = []
    kernelstats_ldp = []
    
    for d in dk:
        
        flop_metrics_frame_tailored = flop_metrics_frame.query('K==%s and VariedDimension==\"K\"'%d)
        trace_frame_tailored = trace_frame.query('K==%s and VariedDimension==\"K\"'%d)
#         pdb.set_trace()
        layer_stats, kern_stats, dictionary = get_stats(flop_metrics_frame_tailored, trace_frame_tailored, layers)
        layer_stats['K'] = [d]*len(layer_stats)
        kern_stats['K'] = [d]*len(kern_stats)
        
        layerstats_ldp.append(layer_stats)
        kernelstats_ldp.append(kern_stats)
    
    layerstats_df = pd.concat(layerstats_ldp)
    kernelstats_df = pd.concat(kernelstats_ldp)
    
    return layerstats_df, kernelstats_df

def scatter(layer_stats, data_labels, data_label_size, xaxis_lbl, yaxis_lbl):
    roofline = [(layer_stats.iloc[i].arithmetic_intensity, layer_stats.iloc[i].throughput) for i in range(len(layer_stats))]
#     mnist_lbls = ['L%s::Bs%s'%(i[0],i[1]) for i in mnist_label_tuples]
    labels = hv.Labels({(xaxis_lbl, yaxis_lbl): roofline, 'text': data_labels}, [xaxis_lbl, yaxis_lbl], 'text')
    labels.opts(opts.Labels(text_font_size=data_label_size, yoffset=0.015))
    scatter = hv.Scatter(roofline)
    scatter.opts(cmap='cool', size=10, alpha=.25)
    plot = labels*scatter
    return plot

