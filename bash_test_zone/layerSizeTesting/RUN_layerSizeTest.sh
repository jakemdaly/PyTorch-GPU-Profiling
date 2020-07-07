#!/bin/bash
ntwrk=mnist
export ntwrk
scp -r -i /home/jakemdaly/Vault/jakedaly2pem.pem /home/jakemdaly/Desktop/bash_test_zone/layerSizeTesting/TorchDCNNLayers.py   rabbit@67.58.52.218:/home/rabbit/Documents/PRL/FPGA_DeconvAcc/mnist/tensorflow/
ssh -i /home/jakemdaly/Vault/jakedaly2pem.pem rabbit@67.58.52.218 'cd /home/rabbit/Documents/PRL/FPGA_DeconvAcc/mnist/tensorflow/; bash '< layerSizeTest.sh
mkdir /home/jakemdaly/Documents/UCSD/Research/gpu/landing_zone/
scp -r -i /home/jakemdaly/Vault/jakedaly2pem.pem rabbit@67.58.52.218:/home/rabbit/Documents/PRL/FPGA_DeconvAcc/mnist/tensorflow/layerSizeTesting/${ntwrk} /home/jakemdaly/Documents/UCSD/Research/gpu/landing_zone/