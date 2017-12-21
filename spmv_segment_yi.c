#include "genresult.cuh"
#include <sys/time.h>
#include <math.h>
#include "mmio.h"
#include "spmv_optimize.c"



MatrixInfo * preProcess(MatrixInfo * mat){
	int M, N, nz;   //M is row number, N is column number and nz is the number of entry
	int *rIndex, *cIndex, *count;
	float *val;

	nz = mat->nz;
	M = mat->M;
	N = mat->N;

    count = (int *)malloc(M * sizeof(int));
	rIndex = (int *)malloc(nz * sizeof(int));
	cIndex = (int *)malloc(nz * sizeof(int));
	val = (float *)malloc(nz * sizeof(float));

    if (count == NULL || rIndex == NULL || cIndex == NULL || val == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors in preprocess!\n");
        exit(EXIT_FAILURE);
    }

    memset(count, 0, M * sizeof(int));

    for (int i=0; i<nz; i++) {
    	count[mat->rIndex[i]]++;
    }


    int pre = count[0];
    count[0] = 0;
    for (int i=1; i<M; i++) {
    	int tmp = count[i];
    	count[i] = count[i-1] + pre;
    	pre = tmp;
    }

    for (int i=0; i<nz; i++) {
    	int row = mat->rIndex[i];
    	int index = count[row]++;
    	rIndex[index] = row;
    	cIndex[index] = mat->cIndex[i];
    	val[index] = mat->val[i];
    }

	MatrixInfo * mat_inf = (MatrixInfo *) malloc(sizeof(MatrixInfo));
    mat_inf->M = M;
    mat_inf->N = N;
    mat_inf->nz = nz;
    mat_inf->rIndex = rIndex;
    mat_inf->cIndex = cIndex;
    mat_inf->val = val;

    return mat_inf;
}


MatrixInfo * copyVector(MatrixInfo * vec){
    MatrixInfo * newVec = (MatrixInfo *) malloc(sizeof(MatrixInfo));

    int M, N, nz;   //M is row number, N is column number and nz is the number of entry
    int *rIndex, *cIndex;
    float *val;

    nz = vec->nz;
    M = vec->M;
    N = vec->N;

    rIndex = NULL;
    cIndex = NULL;
    val = (float *)malloc(vec->nz * sizeof(float));

    memcpy(val, vec->val, vec->nz * sizeof(float)); // ??? size

    newVec->M = M;
    newVec->N = N;
    newVec->nz = nz;
    newVec->rIndex = rIndex;
    newVec->cIndex = cIndex;
    newVec->val = val;
    return newVec;


}

__global__ void multi_kernel( int * r, int * c, float * val, float * v, int nz, int M, int threadLoad, int threadLoadVector, float * res, int blockSize){
    /*Put your kernel(s) implementation here, you don't have to use exactly the
 * same kernel name */

	extern __shared__ int tmp[];
	float *valTemp = (float*)tmp;                       
    int *rowTemp = (int*)&valTemp[blockSize];

	int warp = threadIdx.x / 32;
	int threadStart = (threadIdx.x & 31) + (blockIdx.x * blockDim.x + warp*32 )* threadLoad;
    __syncthreads();

	int i = 0;
	__syncthreads();
    
	for (i=0; i<threadLoad; i++) {

        __syncthreads();

		if (threadStart+i*32 < nz) {
			//printf("ini shared memory... threadIdx: %d, blockIdx: %d, index: %d \n ", threadIdx.x, blockIdx.x, threadStart+i*32);

			valTemp[threadIdx.x] = val[threadStart+i*32] * v[c[threadStart+i*32 ]]; 
			//if (threadStart+i*32==0) printf(" 0th row, 0th col, val=%f, v=%f, valTemp=%f \n",val[threadStart+i*32], v[c[threadStart+i*32 ]], valTemp[threadIdx.x] );
			rowTemp[threadIdx.x] = r[threadStart+i*32];
		    //if (threadStart+i*32==0) printf(" 0th row, 0th col, val=%f, v=%f, valTemp=%f, rowTemp=%d \n",val[threadStart+i*32], v[c[threadStart+i*32 ]], valTemp[threadIdx.x], rowTemp[threadIdx.x] );

		}
		else {
            valTemp[threadIdx.x] = 0; 
			//if (threadStart+i*32==0) printf(" 0th row, 0th col, val=%f, v=%f, valTemp=%f \n",val[threadStart+i*32], v[c[threadStart+i*32 ]], valTemp[threadIdx.x] );
			rowTemp[threadIdx.x] = 0;
		}

		//if (threadStart+i*32==0) printf(" 0th row, 0th col, val=%f, v=%f, valTemp=%f \n",val[threadStart+i*32], v[c[threadStart+i*32 ]], valTemp[threadIdx.x] );
		__syncthreads();

		int lane = threadIdx.x & 31; 
		if ( lane >=1 && rowTemp[threadIdx.x] == rowTemp[threadIdx.x - 1] ) valTemp[threadIdx.x] += valTemp[threadIdx.x-1];
		if ( lane >=2 && rowTemp[threadIdx.x] == rowTemp[threadIdx.x - 2] ) valTemp[threadIdx.x] += valTemp[threadIdx.x-2];
		if ( lane >=4 && rowTemp[threadIdx.x] == rowTemp[threadIdx.x - 4] ) valTemp[threadIdx.x] += valTemp[threadIdx.x-4];
		if ( lane >=8 && rowTemp[threadIdx.x] == rowTemp[threadIdx.x - 8] ) valTemp[threadIdx.x] += valTemp[threadIdx.x-8];
		if ( lane >=16 && rowTemp[threadIdx.x] == rowTemp[threadIdx.x - 16] ) valTemp[threadIdx.x] += valTemp[threadIdx.x-16];
        
        __syncthreads();
		if ( lane==31 || (rowTemp[threadIdx.x+1] != rowTemp[threadIdx.x] ) ) {
			atomicAdd(&res[ rowTemp[threadIdx.x] ],valTemp[threadIdx.x] );
		}
		__syncthreads();


	}

}

void getMulScan(OptAlgType alg, MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    /*Allocate things...*/

    int nDevices;
    cudaGetDeviceCount(&nDevices);
    char * deviceName;
    if (nDevices>0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        deviceName = prop.name;
    }
    else {
    	memcpy(deviceName, "deviceName", 10);
    	deviceName[10] = '\0';
    }




    struct timespec start, end;

    /*Invoke kernel(s)*/
    printf("Start segment scan ... CUDA kernel launch with %d blocks of %d threads ... \n", blockNum, blockSize);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    MatrixInfo * newMat = preProcess(mat);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Pre-processing Time: %lu milli-seconds\n", 1000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000);

    int threadLoad = ceil(newMat->nz*1.0/(blockSize*blockNum));
    int threadLoadVector = ceil(newMat->N*1.0/(blockSize*blockNum));

    MatrixInfo * newVec = copyVector(vec);

    int * mapX, *mapY;
    mapX = (int *)malloc(newVec->nz * sizeof(int));
    mapY = (int *)malloc(newMat->M * sizeof(int));
    memset(mapX, -1, newVec->nz * sizeof(int));
    memset(mapY, -1, newMat->M * sizeof(int));

    //printf("location 1 \n");

    if (alg!=NONE) {
        reorderMapping(newMat, newVec, alg, mapX, mapY, blockNum * blockSize / 32);
        reorderRow(newMat, mapY);
        reorderCol(newMat, mapX);
        transformToNew(newVec, mapX);
    }



    size_t sizeInt = newMat->nz * sizeof(int);
    size_t sizeFlo = newMat->nz * sizeof(float);
    size_t sizeVect = newMat->N * sizeof(float);
    size_t sizeRes = newMat->M * sizeof(float);
    cudaError_t err = cudaSuccess;

    int *d_rows = NULL;
    err = cudaMalloc((void **)&d_rows, sizeInt);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device rows (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int *d_cols = NULL;
    err = cudaMalloc((void **)&d_cols, sizeInt);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device cols (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_vals = NULL;
    err = cudaMalloc((void **)&d_vals, sizeFlo);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vals  (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_vector = NULL;
    err = cudaMalloc((void **)&d_vector, sizeVect);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector  (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_res = NULL;
    err = cudaMalloc((void **)&d_res, sizeRes);
    cudaMemset(d_res,0, sizeRes);  //    memset(count, 0, M);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device res  (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //printf("Copying input data from the host memory to the CUDA device...\n");

    err = cudaMemcpy(d_rows, newMat->rIndex, sizeInt, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy rows from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_cols, newMat->cIndex, sizeInt, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy cols from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_vals, newMat->val, sizeFlo, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vals from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_vector, newVec->val, sizeVect, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//-----------------------------------------------------------------------------------------------------------------------------
    
    // printf("test1...\n");
    // for (int i=0; i<newMat->nz; i++) {
    //     printf("%d %d %f \n",newMat->rIndex[i], newMat->cIndex[i], newMat->val[i]);
    // }

    // for ( int i=0; i<vec->nz; i++) {
    //     printf(" %d th ele : %f \n", i, newVec->val[i]);
    // }


    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    multi_kernel<<<blockNum, blockSize, (blockSize*sizeof(int)+blockSize*sizeof(float))>>>(d_rows, d_cols, d_vals, d_vector, newMat->nz, newMat->M, threadLoad, threadLoadVector, d_res, blockSize);
    cudaDeviceSynchronize(); // this code has to be kept to ensure that all the kernels invoked finish their work
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    
    //printf("location 1 ...");

    printf("The total kernel running time on GPU [%s] is %lu milli-seconds\n", deviceName ,1000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000);
    
    //printf("location 2 ...\n");

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //printf("Copying output data from the CUDA device to the host memory...\n");
    err = cudaMemcpy(res->val, d_res, sizeRes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy result from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //printf("post process...\n");

    if (alg!=NONE) {
        transformToOri(res, mapY);
    }
    //printf("end post process...\n");


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //printf("Free data ...\n");
    freeMatrixInfo(newMat);
    //printf("location 3 ...\n");
    //freeMatrixInfo(newVec);
    //printf("location 4 ...\n");


    err = cudaFree(d_rows);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device rows (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_cols);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device cols (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

        err = cudaFree(d_vals);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vals (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

        err = cudaFree(d_vector);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    



    /*Deallocate, please*/
}
