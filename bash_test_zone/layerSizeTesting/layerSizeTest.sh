#!/bin/bash

source venv/tensorflow/bin/activate
ntwrk=mnist
mkdir layerSizeTesting
mkdir layerSizeTesting/${ntwrk}


##############################
## Layer Testing
##############################

# # Test 1: Varying input channel depth:
# mkdir layerSizeTesting/${ntwrk}/l1/varyInputDim
# # for layer in 1 2 3
# for layer in 1
# do
# 	for dim in 1 2 5 10 20 50 100 200 500
# 	# for bs in 1
# 	do
# 		sudo -HE env PATH=$PATH PYTHONPATH=$PYTHONPATH /usr/local/cuda-10.0/bin/nvprof --metrics achieved_occupancy,ipc,l2_read_throughput,l2_write_throughput,sm_efficiency,flop_count_dp,flop_count_dp_add,flop_count_dp_fma,flop_count_dp_mul,flop_count_hp,flop_count_hp_add,flop_count_hp_fma,flop_count_hp_mul,flop_count_sp,flop_count_sp_add,flop_count_sp_fma,flop_count_sp_mul,flop_count_sp_special,flop_dp_efficiency,flop_hp_efficiency,flop_sp_efficiency --csv --log-file InputDepthFLOPS_l${layer}_size${dim}_${i}.csv python TorchDCNN.py $ntwrk $layer 1 $dim 32 4 
# 		# for i in {1..50}
# 		for i in 1
# 		do
# 			sudo -HE env PATH=$PATH PYTHONPATH=$PYTHONPATH /usr/local/cuda-10.0/bin/nvprof --log-file InputDepthTrace_l${layer}_size${dim}_${i}.csv --print-gpu-trace --csv python TorchDCNN.py $ntwrk $layer 1 $dim 32 4 
# 		done
# 	done
# 	mv *.csv layerSizeTesting/${ntwrk}/l$layer/varyInputDim
# 	rm *.csv -f
# done


# Test 2: Varying output channel depth:
# for layer in 1 2 3
for layer in 2
do
	mkdir layerSizeTesting/${ntwrk}/l${layer}
	mkdir layerSizeTesting/${ntwrk}/l${layer}/varyOutputDim
	for dim in 32 64 128 256 512 1024 2048
	# for dim in 1
	do
		sudo -HE env PATH=$PATH PYTHONPATH=$PYTHONPATH /usr/local/cuda-10.0/bin/nvprof --metrics achieved_occupancy,ipc,l2_read_throughput,l2_write_throughput,sm_efficiency,flop_count_dp,flop_count_dp_add,flop_count_dp_fma,flop_count_dp_mul,flop_count_hp,flop_count_hp_add,flop_count_hp_fma,flop_count_hp_mul,flop_count_sp,flop_count_sp_add,flop_count_sp_fma,flop_count_sp_mul,flop_count_sp_special,flop_dp_efficiency,flop_hp_efficiency,flop_sp_efficiency --csv --log-file OutputDepthFLOPS_l${layer}_size${dim}.csv python TorchDCNNLayers.py $ntwrk $layer 1 32 $dim 6 
		for i in {1..50}
		# for i in 1
		do
			sudo -HE env PATH=$PATH PYTHONPATH=$PYTHONPATH /usr/local/cuda-10.0/bin/nvprof --log-file OutputDepthTrace_l${layer}_size${dim}_${i}.csv --print-gpu-trace --csv python TorchDCNNLayers.py $ntwrk $layer 1 32 $dim 6
		done
	done
	mv *.csv layerSizeTesting/${ntwrk}/l${layer}/varyOutputDim
	rm *.csv -f
done



# # Test 3: Varying filter size:
# mkdir layerSizeTesting/${ntwrk}/l1/varyK
# # for layer in 1 2 3
# for layer in 1
# do
# 	for dim in 2 4 6 8 10 12 14 16
# 	# for bs in 1
# 	do
# 		sudo -HE env PATH=$PATH PYTHONPATH=$PYTHONPATH /usr/local/cuda-10.0/bin/nvprof --metrics achieved_occupancy,ipc,l2_read_throughput,l2_write_throughput,sm_efficiency,flop_count_dp,flop_count_dp_add,flop_count_dp_fma,flop_count_dp_mul,flop_count_hp,flop_count_hp_add,flop_count_hp_fma,flop_count_hp_mul,flop_count_sp,flop_count_sp_add,flop_count_sp_fma,flop_count_sp_mul,flop_count_sp_special,flop_dp_efficiency,flop_hp_efficiency,flop_sp_efficiency --csv --log-file KFLOPS_l${layer}_size${dim}_${i}.csv python TorchDCNN.py $ntwrk $layer 1 10 32 $dim 
# 		# for i in {1..50}
# 		for i in 1
# 		do
# 			sudo -HE env PATH=$PATH PYTHONPATH=$PYTHONPATH /usr/local/cuda-10.0/bin/nvprof --log-file KTrace_l${layer}_size${dim}_${i}.csv --print-gpu-trace --csv python TorchDCNN.py $ntwrk $layer 1 10 32 $dim
# 		done
# 	done
# 	mv *.csv layerSizeTesting/${ntwrk}/l$layer/varyK
# 	rm *.csv -f
# done