#include "genresult.cuh"
#include <sys/time.h>

/* Put your own kernel(s) here*/

void getMulDesign(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    /*Allocate*/

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    /*Your own magic here!*/


	
    //cudaDeviceSynchronize(); // this line is needed for the completeness of the program
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Your Own Kernel Time: %lu milli-seconds\n", 1000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000);

    /*Deallocate*/
}
