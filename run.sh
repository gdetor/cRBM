#!/bin/bash
#

make clean
make
./bin/rbm 3000
cd ./data
python plot_rfs.py weights_rbm.dat
cd ..
