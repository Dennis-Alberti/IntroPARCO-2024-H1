Here there are the information to reproducing the results:

The code is one so to obtain the different results is needed to comment and uncomment the the desired parts.

For compilation and running operation I use 4 different .pbs
1. serial.pbs for running the serial
2. implicit.pbs for running the implicit
3. openmp.pbs for running the OpenMP and the bandwidth
4. speedup_efficency for running the Speedup and the efficency

The result all printed in the relative .cvs files

Processor: Intel(R) Core(TM) i7-8565U CPU 1.80Hz, 4 cores, 8 logical processors.
RAM: 8GB DDR4.
Operating System: Windows 11
Compiler: GCC 13.2.0, G++ 13.2.0 (MinGW Compiler), GCC 9.1.0 on the cluster.
Libraries: 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>
#include <stdbool.h>
