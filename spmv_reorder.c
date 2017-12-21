#include "genresult.cuh"
#include <sys/time.h>
#include <cmath>
#include <assert.h>
#include <cuda.h>
#include <mmio.h>
#include "utilsLinkedList.c"
#include "spmv_optimize_graph.c"
//#include "hashmap.c"

#define WARP_SIZE 32

//typedef enum {ALG_SEGMENT = 0, ALG_DESIGN = 1, ALG_FTP = 2, ALG_GRAPH = 3} AlgType;

/*
Reorder the input matrix so that elements in the same row be placed consecutively in MatrixInfo.

int preprocess(MatrixInfo * matrix){
    printf("111111111111111111111a");
    int nz = matrix->nz, m = matrix->M;
    int * rIndex = matrix->rIndex, * cIndex = matrix->cIndex;
    float * val = matrix->val;
    int * eleInRow = (int*)malloc(m*sizeof(int));
    memset(eleInRow, 0, m);
    for(int i=0;i<nz;i++){
        eleInRow[rIndex[i]]++;
    }
    int * eleInRowAcc = (int*)malloc(m*sizeof(int));
    for(int i=0, temp=0;i<m;i++){
        temp = temp+eleInRow[i];
        eleInRowAcc[i] = temp;
    }
    printf("2222222222222222222a");
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
    printf("33333333333333333a");
    matrix->rIndex = rIndexNew;
    matrix->cIndex = cIndexNew;
    matrix->val = valNew;
    free(rIndex);
    free(cIndex);
    free(val);
    return 1;
}*/

/*
Get the reorderX indices used to reorderX input matrix and vector, by first touch packing algorithm.
*/
void getReorder_ftp(int * reorderX, MatrixInfo * matrix, int n, int nz){
    int * cIndex = matrix->cIndex;
    int * tag = (int*)malloc(n*sizeof(int));
    memset(tag, 0, n*sizeof(int));
    int nextAvailable = 0;
    for(int i=0;i<nz;i++){
        if(tag[cIndex[i]]==0){
            tag[cIndex[i]] = 1;
            reorderX[nextAvailable++] = cIndex[i];
            //reorder_reverseX[cIndex[i]] = nextAvailable++;
        }
    }
}

void getReorder_graph_new(MatrixInfo * mat, MatrixInfo * vec, int * reorderX, int * reorderY, int warpNum){
    graphPacking(mat, vec, reorderX, reorderY, warpNum);
}

/*
Get the reorderX indices used to reorderX input matrix and vector, by graph based packing algorithm.
TODO. input parameter should be n, not m.
*/
void getReorder_graph(int * reorderX, int * reorder_reverseX, MatrixInfo * matrix, int m, int nz, int elePerWarp){
    int ele_per_cache_line = 64/sizeof(int);
    printf("ele_per_cache_line: %d. elePerWarp: %d. \n", ele_per_cache_line, elePerWarp);
    Graph * graph = initGraph(m);
    generateGraph(matrix->cIndex, graph, elePerWarp, nz);
    int index = 0, count = 0;
    int * chosen = (int*)malloc(m*sizeof(int));     // Simple hashset impl, 1 means existed, 0 means nonexisted in the chosen set.
    int * tag = (int*)malloc(m*sizeof(int));
    memset(tag, 0, m*sizeof(int));                  // Tag of whether this item has been added into the reordered vector.
    int isNewCacheLine = 1;                         // a flag of whether we are starting fulfilling a new cache line.
    while(index < m){
        if(count == ele_per_cache_line){            // If current cache line has been fulfilled, initialize the chosen.
            memset(chosen, 0, m*sizeof(int));
            count=0;
            isNewCacheLine = 1;
        }
        int a = -1, b = -1;
        //printf("graph before getting max edge...\n");
        //printGraph(graph);
        getMaxEdgeAndRemove(graph, &a, &b, chosen, isNewCacheLine);         // get the edge with maximum weight, within those connected to the chosen vertices.
        //removeEdge(graph, a, b);
        //printf("getting from getMaxEdge - (%d, %d).\n", a, b);
        //printf("graph after getting max edge...\n");
        //printGraph(graph);
        //exit(0);
        if(a != -1 && b != -1 && a!=b){
            if(tag[a] == 0){
                reorderX[index] = a;
                reorder_reverseX[a] = index++;
                tag[a] = 1;
                count++;
            }
            if(tag[b] == 0){
                reorderX[index] = b;
                reorder_reverseX[b] = index++;
                tag[b] = 1;
                count++;
            }
        }
        else
            printf("ERROR from getMaxEdge, a = %d, b = %d.\n", a, b);

        isNewCacheLine = 0;
        chosen[a] = 1;
        chosen[b] = 1;
        if(count > ele_per_cache_line){
            printf("ERROR - count = %d, ele_per_cache_line = %d. \n", count, ele_per_cache_line);
            exit(0);
        }

/*
        if(count < ele_per_cache_line){             // If current cache line hasn't been fulfilled, switch isNewCacheLine to false(0), adding a and b into chosen.
            isNewCacheLine = 0;
            chosen[a] = 1;
            chosen[b] = 1;
        } else if(count == ele_per_cache_line){     // If current cache line has been fulfilled, initialize the chosen.
            memset(chosen, 0, m*sizeof(int));
            count=0;
            isNewCacheLine = 1;
        } else{                                     // count > ele_per_cache_line, this situation should not be appearing.
            printf("ERROR - count = %d, ele_per_cache_line = %d. \n", count, ele_per_cache_line);
        }*/
    }

}




void transformToNewNew(MatrixInfo * vec, MatrixInfo * vec_trans, int * reorderX){
    //printf("start transforming to new vector X ...\n");
    for(int i=0;i<vec->nz;i++){
        vec_trans->val[i] = vec->val[reorderX[i]];
    }
}

void reorderCol(MatrixInfo * mat, int * reorder_reverseX){
    //printf("start reordering column ...\n");
    // printf("%d \n", mat->cIndex[0]);
    // printf("%d \n", mapX[0]);
    for ( int i = 0; i < mat->nz; i++ ) {
        //printf("before - i=%d, mat->cIndex[i] = %d.\n", i, mat->cIndex[i]);
        mat->cIndex[i] = reorder_reverseX[mat->cIndex[i]];
        //printf("after - i=%d, mat->cIndex[i] = %d.\n", i, mat->cIndex[i]);
        //printf("%d th iteration, location : %d, value : %d", i,);
    }
}

void reorderRow(MatrixInfo * mat, int * reorder_reverseY){
    //printf("start reorderRow ...\n");
    for ( int i = 0; i < mat->nz; i++ ) {
        mat->rIndex[i] = reorder_reverseY[mat->rIndex[i]];
        //printf("%d th iteration, location : %d, value : %d", i,);
    }
}

/*
Transform input vector and result into new vectors, and col to col_trans, based on reorderX vector.
*/
void transformToNew(MatrixInfo * vec, MatrixInfo * vec_trans, MatrixInfo * res, MatrixInfo * res_trans, int * col, int * col_trans, int nz, int * reorderX, int * reorder_reverseX){
    float * vec_val = vec->val;
    float * vec_trans_val = vec_trans->val;
    for(int i=0;i<vec->M;i++){
        vec_trans_val[i] = vec_val[reorderX[i]];
    }
    for(int i=0;i<nz;i++){
        col_trans[i] = reorder_reverseX[col[i]];
    }
}

void transformToOri(MatrixInfo * res_trans, MatrixInfo * res, int * reorderY, int algType){
    if(algType==0){
        for(int i=0;i<res->nz;i++){
            res->val[i] = res_trans->val[i];
        }
    } else if(algType==1){
        for(int i=0;i<res->nz;i++){
            res->val[reorderY[i]] = res_trans->val[i];
        }
    } else
        exit(0);
}

void printVector(int * vector, int l, const char * name){
    printf("\nprinting %s.\n", name);
    for(int i=0;i<l;i++){
        printf("%d, ", vector[i]);
    }
}

void printFloatVector(float * vector, int l, const char * name){
    printf("\nprinting %s.\n", name);
    for(int i=0;i<l;i++){
        printf("%f, ", vector[i]);
    }
}

MatrixInfo * copyMatrix(MatrixInfo * from){
    MatrixInfo * newVec = (MatrixInfo *) malloc(sizeof(MatrixInfo));

    int M, N, nz;   //M is row number, N is column number and nz is the number of entry
    int *rIndex, *cIndex;
    float *val;

    nz = from->nz;
    M = from->M;
    N = from->N;

    if(from->rIndex == NULL && from->cIndex==NULL){    // meaning it's a vector
        rIndex = NULL;
        cIndex = NULL;
    } else{
        rIndex = (int *)malloc(from->nz * sizeof(int));
        cIndex = (int *)malloc(from->nz * sizeof(int));
        memcpy(rIndex, from->rIndex, from->nz * sizeof(int));
        memcpy(cIndex, from->cIndex, from->nz * sizeof(int));
    }

    val = (float *)malloc(from->nz * sizeof(float));

    memcpy(val, from->val, from->nz * sizeof(float)); // ??? size

    newVec->M = M;
    newVec->N = N;
    newVec->nz = nz;
    newVec->rIndex = rIndex;
    newVec->cIndex = cIndex;
    newVec->val = val;
    return newVec;


}

void reverse(int * reorderX, int * reorder_reverseX, int n, int * reorderY, int * reorder_reverseY, int m){
    for(int i=0;i<n;i++){
        reorder_reverseX[reorderX[i]] = i;
    }
    for(int i=0;i<m;i++){
        reorder_reverseY[reorderY[i]] = i;
    }
}


void spmv_segmentedScan(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    struct timespec start, end;


    /*Allocate things...*/
    int nz = mat->nz, m = mat->M, n = mat->N;
    int warps = blockSize / WARP_SIZE * blockNum;
    //int warps = blockSize / WARP_SIZE * i;
    int elePerWarp = (nz-1)/warps+1;  // # eles/warp = the ceiling of nz over warps, except for the last warp.
    //printf("@#@#@#@ elePerWarp: %d\n", elePerWarp);
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

void getMulReorder(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum, int algType){
    /*Allocate*/


    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    if(preprocess(mat)!=1){
        printf("Preprocess failed!\n");
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("### Preprocessing Time: %lu milli-seconds\n\n", 1000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000);
    /*Your own magic here!*/

    int nz = mat->nz, m = mat->M, n = mat->N;
    //int warps = blockSize / WARP_SIZE * blockNum;
    //int elePerWarp = (nz-1)/warps+1;  // # eles/warp = the ceiling of nz over warps, except for the last warp.

    int * reorderX = (int*)malloc(n*sizeof(int));
    int * reorder_reverseX = (int*)malloc(n*sizeof(int));
    memset(reorderX, 0, n*sizeof(int));
    memset(reorder_reverseX, 0, n*sizeof(int));
    int * reorderY = (int*)malloc(m*sizeof(int));
    int * reorder_reverseY = (int*)malloc(m*sizeof(int));
    memset(reorderY, 0, m*sizeof(int));
    memset(reorder_reverseY, 0, m*sizeof(int));

    printf("Getting reordering mapping......\n");
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    switch(algType){
        case 0:  // FTP Algorithm
            getReorder_ftp(reorderX, mat, n, nz);
            break;
        case 1:  // Graph based packing algorithm
            getReorder_graph_new(mat, vec, reorderX, reorderY, blockNum * blockSize / WARP_SIZE);
            break;
        default:
            break;
    }
    reverse(reorderX, reorder_reverseX, n, reorderY, reorder_reverseY, m);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("### Getting reordering time: %lu seconds\n", end.tv_sec - start.tv_sec);
    //printVector(reorder_reverseX, m, "reorder_reverseX");
    //exit(0);
    //printVector(mat->cIndex, nz, "col");
    MatrixInfo * mat_trans = copyMatrix(mat);
    //printVector(mat_trans->cIndex, nz, "col");
    printf("Reordering column......\n");
    reorderCol(mat_trans, reorder_reverseX);
    printf("Reordering row......\n");
    if(algType==1)
        reorderRow(mat_trans, reorder_reverseY);

    MatrixInfo * vec_trans = copyMatrix(vec);
    MatrixInfo * res_trans = copyMatrix(res);
    printf("Transforming input vector......\n");
    transformToNewNew(vec, vec_trans, reorderX);

    //printVector(mat_trans->cIndex, nz, "col_trans");

    printf("\nFinish preparation.\n\nDoing segmented scan......\n");

    //int * col_trans = (int*)malloc(nz*sizeof(int));

    //transformToNew(vec, vec_trans, res, res_trans, mat->cIndex, col_trans, nz, reorderX, reorder_reverseX);

    //mat_trans->cIndex = col_trans;

    // printFloatVector(mat_trans->val, nz, "matrix val");
    // printVector(mat_trans->rIndex, nz, "rIndex");
    // printVector(mat_trans->cIndex, nz, "col");
    // printFloatVector(vec_trans->val, n, "col_trans");
    // //printVector(reorderX, m, "reorderX");
    // printVector(reorder_reverseX, n, "reorder_reverseX");
    // printFloatVector(vec_trans->val, n, "vec_trans");
    // exit(0);

    spmv_segmentedScan(mat_trans, vec_trans, res_trans, blockSize, blockNum);

    //printf("Segmented")

    transformToOri(res_trans, res, reorderY, algType);


    //printFloatVector(res->val, m, "res");
    //printFloatVector(res_trans->val, m, "res_trans");

    //cudaDeviceSynchronize(); // this line is needed for the completeness of the program
    //clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    //printf("Your Own Kernel Time: %lu milli-seconds\n", 1000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000);

    /*Deallocate*/

}
