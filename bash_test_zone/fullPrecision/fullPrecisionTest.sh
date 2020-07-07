#!/bin/bash

source venv/tensorflow/bin/activate
ntwrk=mnist
mkdir fullPrecision
mkdir fullPrecision/${ntwrk}
for layer in 1
# for layer in 1 2 3 4 5
do
	mkdir fullPrecision/${ntwrk}/l${layer}
	# for bs in 1 2 5 10 20 50 100 200 500 1000 2000 5000 10000
	for bs in 1
	do
		sudo -HE env PATH=$PATH PYTHONPATH=$PYTHONPATH /usr/local/cuda-10.0/bin/nvprof --metrics achieved_occupancy,ipc,l2_read_throughput,l2_write_throughput,sm_efficiency,flop_count_dp,flop_count_dp_add,flop_count_dp_fma,flop_count_dp_mul,flop_count_hp,flop_count_hp_add,flop_count_hp_fma,flop_count_hp_mul,flop_count_sp,flop_count_sp_add,flop_count_sp_fma,flop_count_sp_mul,flop_count_sp_special,flop_dp_efficiency,flop_hp_efficiency,flop_sp_efficiency --csv --log-file fullPrecisionFLOPS_l${layer}_bs${bs}.csv python TorchDCNNFull.py $ntwrk $layer $bs
		# for i in {1..50}
		for i in 1
		do
			sudo -HE env PATH=$PATH PYTHONPATH=$PYTHONPATH /usr/local/cuda-10.0/bin/nvprof --log-file fullPrecisionTrace_l${layer}_bs${bs}_${i}.csv --print-gpu-trace --csv python TorchDCNNFull.py $ntwrk $layer $bs
		done
	done
	mv *.csv fullPrecision/${ntwrk}/l${layer}
	rm *.csv -f
done