# cRBM

This is a C implementation of a Restricted Bolzmann Machine [1]. In addition, we have implemented a Non-negative Matrix 
Factorization algorithm based on the RBM implementation (see [2] for more details). 

In the current implementation we make use of the PCG random number generator [3]. And can be found here: http://www.pcg-random.org/

This branch contains: 
```
├── bin
├── data
│   └── plot_rfs.py  -- Plot the receptive fields
├── include
│   ├── pcg_basic.h  -- Random Number Generator
│   └── rbm.h
├── LICENSE
├── Makefile
├── obj
├── README.md
├── run.sh
└── src
    ├── examples.c
    ├── functions.c
    ├── load_data.c
    ├── main.c
    ├── pcg_basic.c
    └── rbm.c
```


## Platform Information
Linux #1 SMP PREEMPT Wed Jul 5 18:23:08 CEST 2017 x86_64 GNU/Linux

## Hardware Information
model name	: Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz
vendor_id	: GenuineIntel

## Compiler Information
gcc (GCC) 7.1.1 20170630


References
==========

1. Paul Smolensky, "Information processing in dynamical systems: Foundations of harmony theory", 1986.

2. Nguyen Tu Dinh, Tran Truyen, Phung Dinh and Venkatesh Svetha, "Learning parts-based representations with nonnegative restricted boltzmann machine", Asian Conference on Machine Learning, 133--148, 2013.

3. Melissa E. O'Neill, "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation",
Harvey Mudd College, Claremont, CA, HMC-CS-2014-0905, 2014
