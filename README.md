# openfhe_apu
Testing porting OpenFHE to Accelerated Processing Unit (APU)

# Description
The idea, initially described in [1], is to port the OpenFHE's [2] most computationally expensive function with the most overhead to the GPU devices. In this particular case, 
AMD MI300A Accelerated Processing Unit (APU) [3] is being used. The function ApproxSwitchCRTBasis was modified to be ran on the GPU part of the APU.
The main effort was to represent 128 bit integers as 2 64 bit integers placed in the adjacent locations in an array. The rest of the dependent 
functions have been rewritten as well.

# Installation
The modified ApproxSwitchCRTBasis function can replace existing one in the `src/core/include/lattice/hal/default/dcrtpoly-impl.h` file. In addition `apu.h` file should be placed in
the same folder and included in the `dcrtpoly-impl.h`. The Encryption algorithm is the BFV (Brakerski/Fan-Vercauteren). The CryptoContext parameters are included in the `simple-integers.cpp` file

# Performance
The functions is rewritten and tested in the scenario where ring dimension is 32768 and sizeP
and sizeQ are 8. The rewritten function was compared against the original one from [2] which runs on an AMD EPYC 9374F CPU. 
The performance gain is around 25 % on the APU, not counting data (un)marshalling. 

# References
[1] https://openfhe.discourse.group/t/openfhe-on-amd-apus/2188  
[2] https://github.com/openfheorg/openfhe-development  
[3] https://www.amd.com/en/products/accelerators/instinct/mi300/mi300a.html 
