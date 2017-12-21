#include "genresult.cuh"
#include <sys/time.h>
#include <cmath>
#include <assert.h>
#include <cuda.h>
#include <mmio.h>

#define WARP_SIZE 32

/*
static __inline__ __device__ float atomicAdd(float *addr, float val)
{
    float old=*addr, assumed;

    do {
        assumed = old;
        old = int_as_float( atomicCAS((int*)addr,
                                        float_as_int(assumed),
                                        float_as_int(val+assumed)));
    } while( assumed!=old );

    return old;
}*/

/*
Reorder the input matrix so that elements in the same row be placed consecutively in MatrixInfo.
*/
int preprocess(MatrixInfo * matrix){
    int nz = matrix->nz, m = matrix->M;
    //printf("nz: %d, M: %d, N: %d.\n", nz, m, matrix->N);
    int * rIndex = matrix->rIndex, * cIndex = matrix->cIndex;
    //printf("rIndex[0] = %d, cIndex[0] = %d, val[0] = %f.\n", rIndex[0], cIndex[0], matrix->val[0]);
    float * val = matrix->val;
    int * eleInRow = (int*)malloc(m*sizeof(int));
    memset(eleInRow, 0, m*sizeof(int));
    for(int i=0;i<nz;i++){
        eleInRow[rIndex[i]]++;
    }
    int * eleInRowAcc = (int*)malloc(m*sizeof(int));
    for(int i=0, temp=0;i<m;i++){
        temp = temp+eleInRow[i];
        eleInRowAcc[i] = temp;
    }
    //printf("eleInRowAcc[m-1] = %d, should = nz.\n", eleInRowAcc[m-1]);
    int * rIndexNew = (int*)malloc(nz*sizeof(int));
    int * cIndexNew = (int*)malloc(nz*sizeof(int));
    float * valNew = (float*)malloc(nz*sizeof(float));
    for(int i=0;i<nz;i++){
        int row = rIndex[i];
        int p = eleInRowAcc[row] - eleInRow[row];
        rIndexNew[p] = rIndex[i];
        cIndexNew[p] = cIndex[i];
        valNew[p] = val[i];
        eleInRow[row]--;
    }
    //printf("Here...\n");
    matrix->rIndex = rIndexNew;
    matrix->cIndex = cIndexNew;
    matrix->val = valNew;
    free(rIndex);
    free(cIndex);
    free(val);
    return 1;
}

__global__ void putProduct_kernel(int * cIndex, float * val, float * vector, int nz, int numsPerThread){
    /*Put your kernel(s) implementation here, you don't have to use exactly the
 * same kernel name */
    int thr = threadIdx.x + blockIdx.x * blockDim.x;
    int indexFrom = thr*numsPerThread, indexTo = (thr+1)*numsPerThread;
    for(int i=indexFrom;i<indexTo && i<nz;i++){
        val[i] = val[i]*vector[cIndex[i]];
    }
}

__global__ void segmented_scan(int * rIndex, int * cIndex, float * val, float * vec, float * y, int nz, int elePerWarp, int blockSize, int blockNum){

    extern __shared__ int s[];
    int * shared_rows = s;                                      // The shared memory of rIndex
    float * shared_val = (float*)&shared_rows[blockSize];       // The shared memory of val
    int * last_row = (int*)&shared_val[blockSize];              // The row index of the last element of the last round(32 elements) for the same warp
    float * last_val = (float*)&last_row[blockSize/WARP_SIZE];  // The value of the last element of the last round(32 elements) for the same warp


    int global_threadId = blockSize*blockIdx.x + threadIdx.x;   // Global thread id
    int global_warpId = global_threadId/WARP_SIZE;              // Global warp id, which current thread belongs to
    int thread_lane = threadIdx.x & (WARP_SIZE-1);              // Current thread lane, within the warp
    int warp_lane = threadIdx.x/WARP_SIZE;                      // Current warp lane, within the block

    int warpFrom = global_warpId*elePerWarp;                    // This warp deals with the elements from warpFrom
    int warpTo = min((global_warpId+1)*elePerWarp, nz);         // -- to warpTo, if less than nz
    int indexFrom = warpFrom + thread_lane;                     // This thread deals with the elements from indexFrom
    int indexTo = warpTo;                                       // -- to indexTo, if less than nz, nz otherwise

    if(indexFrom >= indexTo) return;                            // This warp has no work to do

    int first_row = rIndex[elePerWarp * global_warpId];         // The row index of the first element in the warp

    if(thread_lane == 0){                                       // Initialization
        last_row[warp_lane] = first_row;
        last_val[warp_lane] = 0;
    }
    __syncthreads();

    int i=indexFrom;
    for(;i<indexTo;i+=WARP_SIZE){
        shared_rows[threadIdx.x] = rIndex[i];                   // Copy the row index to shared memory
        shared_val[threadIdx.x] = val[i]*vec[cIndex[i]];        //  Take the multiplication and copy the val to shared memory

        if(thread_lane == 0){
            if(shared_rows[threadIdx.x] == last_row[warp_lane])
                shared_val[threadIdx.x] += last_val[warp_lane];
            else if(last_row[warp_lane] != first_row)
                y[last_row[warp_lane]] = last_val[warp_lane];
            else
                atomicAdd(y+last_row[warp_lane], last_val[warp_lane]);
        }
        __syncthreads();

        if( thread_lane >=  1 && shared_rows[threadIdx.x] == shared_rows[threadIdx.x - 1] ) { shared_val[threadIdx.x] += shared_val[threadIdx.x -  1]; __syncthreads(); }
        if( thread_lane >=  2 && shared_rows[threadIdx.x] == shared_rows[threadIdx.x - 2] ) { shared_val[threadIdx.x] += shared_val[threadIdx.x -  2]; __syncthreads(); }
        if( thread_lane >=  4 && shared_rows[threadIdx.x] == shared_rows[threadIdx.x - 4] ) { shared_val[threadIdx.x] += shared_val[threadIdx.x -  4]; __syncthreads(); }
        if( thread_lane >=  8 && shared_rows[threadIdx.x] == shared_rows[threadIdx.x - 8] ) { shared_val[threadIdx.x] += shared_val[threadIdx.x -  8]; __syncthreads(); }
        if( thread_lane >= 16 && shared_rows[threadIdx.x] == shared_rows[threadIdx.x -16] ) { shared_val[threadIdx.x] += shared_val[threadIdx.x - 16]; __syncthreads(); }
        __syncthreads();

        if(thread_lane == WARP_SIZE-1 || i == warpTo-1){
            last_row[warp_lane] = shared_rows[threadIdx.x];
            last_val[warp_lane] = shared_val[threadIdx.x];
        } else if(shared_rows[threadIdx.x] != shared_rows[threadIdx.x+1]){
            if(shared_rows[threadIdx.x]==first_row)
                atomicAdd(y+shared_rows[threadIdx.x], shared_val[threadIdx.x]);
            else
                y[shared_rows[threadIdx.x]] = shared_val[threadIdx.x];
        }
        __syncthreads();
    }

    if(thread_lane == WARP_SIZE-1 || i == warpTo-1)
        atomicAdd(y+last_row[warp_lane], last_val[warp_lane]);
    __syncthreads();
}

void getMulScan(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    struct timespec start, end;
    /*Do the preprocessing...*/
    printf("Preprocessing......\n");
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    if(preprocess(mat)!=1){
        printf("Preprocess failed!\n");
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Preprocessing Time: %lu milli-seconds\n\n", 1000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000);

    /*Allocate things...*/
    int nz = mat->nz, m = mat->M, n = mat->N;
    int warps = blockSize / WARP_SIZE * blockNum;
    int elePerWarp = (nz-1)/warps+1;  // # eles/warp = the ceiling of nz over warps, except for the last warp.
    //int numsPerThread = (nz-1)/(blockSize*blockNum)+1;
    int * d_rIndex, * d_cIndex;
    float * d_val, * d_vector, * d_res;
    cudaMalloc((void **)&d_rIndex, nz*sizeof(int));
    cudaMalloc((void **)&d_cIndex, nz*sizeof(int));
    cudaMalloc((void **)&d_val, nz*sizeof(float));
    cudaMalloc((void **)&d_vector, n*sizeof(float));
    cudaMalloc((void **)&d_res, m*sizeof(float));

    cudaMemcpy(d_rIndex, mat->rIndex, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cIndex, mat->cIndex, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, mat->val, nz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vec->val, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res->val, m*sizeof(float), cudaMemcpyHostToDevice);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    /*Invoke kernel(s)*/

    segmented_scan<<<blockNum, blockSize, (blockSize + blockSize/WARP_SIZE)*(sizeof(int)+sizeof(float))>>>(d_rIndex, d_cIndex, d_val, d_vector, d_res, nz, elePerWarp, blockSize, blockNum);

    cudaDeviceSynchronize(); // this code has to be kept to ensure that all the kernels invoked finish their work
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    int nDevices;
    char deviceName[256];
    cudaGetDeviceCount(&nDevices);
    if(nDevices>=1){
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        memcpy(deviceName, prop.name, strlen(prop.name));
    }

    printf("The total kernel running time on GPU [%s]: %lu milli-seconds\n", deviceName, 1000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000);

    cudaMemcpy(mat->rIndex, d_rIndex, nz*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(mat->cIndex, d_cIndex, nz*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(mat->val, d_val, nz*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(res->val, d_res, m*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_rIndex);
    cudaFree(d_cIndex);
    cudaFree(d_val);
    cudaFree(d_vector);
    cudaFree(d_res);

    /*Deallocate, please*/
}
