#include "genresult.cuh"
#include <sys/time.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include "hashmap.c"

#define BIG_PRIME 49999
#define SMALL_PRIME 11
#define CACHE_LINE_SIZE 32

/* Put your own kernel(s) here*/


LinkedNode ** generateGraph(MatrixInfo * mat, int warpNum, Map * sortEle, int * size, int xORy) {
	//printf("start  generateGraph...\n");
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    /********************************
    generate hash map for each element
    *********************************/
    int warpLoad = ceil(mat->nz *1.0 / warpNum);

    WholeMap * map = createWholeMap(BIG_PRIME);

	int numIteration = ceil (mat-> nz * 1.0 / 32 );



	// for (int i = 0; i<numIteration; i++) {  // for each iteration in a warp
	// 	//if (i% 1000==0 ) printf(" %d th iteration ... \n", i);
	// 	int * cols = (int *)malloc(32 * sizeof(int));
 //        memset(cols, -1, 32 * sizeof(int));

 //        Map * setCol = createMap(31);
	// 	int countCol = 0;

	// 	for (int j = i*32; j< (i+1)*32 && j < mat-> nz; j++) {   // collect the ele in the current iteration

    for (int u=0; u<warpNum; u++) {
        int threadNum = ceil(warpLoad * 1.0 /32);
        for (int v = 0; v< threadNum; v++) {        //if (i% 1000==0 ) printf(" %d th iteration ... \n", i);

            int * cols = (int *)malloc(32 * sizeof(int));
            memset(cols, -1, 32 * sizeof(int));

            Map * setCol = createMap(31);
            int countCol = 0;

            for (int j = u*warpLoad+v*32; j< u*warpLoad+(v+1)*32 && j< mat->nz && j< (u+1)*warpLoad; j++) {
    			int col = -1;
    			if (xORy==1) col = mat->cIndex[j];
    			else col = mat->rIndex[j];
    			int a = getInMap(setCol, col);
    			if (a==-1) {
    				putIntoMap(setCol, col, 1);
    				cols[countCol++] = col;
    			}
    		}

    		free(setCol);

    		for (int j=0; j<31; j++) {  //put those ele into map
    			if (cols[j] ==-1) break;
    			int e1 = cols[j];
    			for (int k=j+1; k<32; k++) {
    				if (cols[k]==-1) break;
    				int e2 = cols[k];

    				Map * tmp = getInWholeMap(map, e1);
    				if (tmp == NULL) {
    					Map * newMap = createMap(SMALL_PRIME);
    					putIntoMap(newMap, e2, 1);
    					putIntoWholeMap(map, e1, newMap);
    				}
    				else {
    					int weight = getInMap(tmp, e2);
                        if(weight==-1)
                            putIntoMap(tmp, e2, 1);
                        else
        					putIntoMap(tmp, e2, weight+1);
    				}

    				tmp = getInWholeMap(map, e2);
    				if (tmp == NULL) {
    					Map * newMap = createMap(SMALL_PRIME);
    					putIntoMap(newMap, e1, 1);
    					putIntoWholeMap(map, e2, newMap);
    				}
    				else {
    					int weight = getInMap(tmp, e1);
                        if(weight==-1)
                            putIntoMap(tmp, e1, 1);
                        else
        					putIntoMap(tmp, e1, weight+1);
    				}


    			}
    		}
    	}
    }
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	//printf("The total generate two dimention map time is %lu sec \n" ,  (end.tv_sec - start.tv_sec) );

	/*****************************
	sort the map and build the graph
	*****************************/

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	LinkedNode ** graphMat = sortWholeGraph(map, sortEle, size);//

	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	//printf("The total generate Graph time is %lu sec \n" ,  (end.tv_sec - start.tv_sec) );
	return graphMat;

}


void generateMapping(LinkedNode ** graphMat, int M, int size, int * mapping, int cacheSize, Map * sortEle) {
    //printf("start generateMapping...\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    int index = 0;
	int * flags = (int*)malloc(M*sizeof(int)); // already in cache line?
	for (int i=0; i<M; i++) {
		flags[i] = 0;
	}

	int count = 0; // how many in the current cache line
	int eleInCache[cacheSize]; // keep a record of elements in current cacheline
	int upperBound = 0;

    while (1) {
    	while (upperBound<size &&  graphMat[upperBound]->next == NULL) {
    		upperBound++;
    	}
    	if (upperBound == size) break;

    	int e1 = graphMat[upperBound]->val;
    	int e2 = graphMat[upperBound]->next ->val;
    	upperBound++;
    	//printf("upperBound %d \n", upperBound);
    	//printf("finf global max %d  %d \n", e1, e2);


        count = count % cacheSize;
        if (count ==0) {
     		for (int i = 0; i< cacheSize; i++) {
	    		eleInCache[i] = -1;
		    }
        }

		while (count < cacheSize ) {
			//printf("now putting");

			if (count < cacheSize && flags[e1]==0) {
				mapping[index++] = e1;
				flags[e1] = 1; // denote already in cache line
				eleInCache[count] = e1; // denote already in current cache line
				//graphMat[e1] -> next = NULL;
				count++;
				 //rintf("putting %d \n", e1);
			}
			if (count < cacheSize && flags[e2]==0) {
				mapping[index++] = e2;
				flags[e2] = 1;
				eleInCache[count] = e2;
				//graphMat[e2] -> next = NULL;
				count++;
				 //printf("putting %d \n", e2);
			}

			// printf("already putting, size: %d\n", count);

			//graphMat[getInMap(sortEle, e1)].next = graphMat[e1].next -> next; // delete
			//printf("delete 1\n");

			LinkedNode * ttmp =  graphMat[getInMap(sortEle, e1)];
			while (ttmp->next!= NULL && ttmp->next->val != e2) ttmp = ttmp-> next;
			if (ttmp -> next != NULL) ttmp -> next = ttmp -> next -> next;


			LinkedNode * tmp =  graphMat[getInMap(sortEle, e2)];
			while (tmp->next!= NULL && tmp->next->val != e1) tmp = tmp-> next;
			if (tmp -> next != NULL) tmp -> next = tmp -> next -> next;

			// printf("delete \n");

    // printf(" after delete printing graph X ...\n");
    // for (int i=0; i<M; i++) {
    //     LinkedNode * tmp = graphMat[i];
    //     while( tmp != NULL ) {
    // 	    printf( "val and weight : %d  %d\n", tmp->val ,tmp -> weight);
    // 	    tmp = tmp->next;
    //     }
    // }

			if (count < cacheSize) { // find local max neighbor
				int localMax = 0;
				for (int i =0; i<cacheSize; i++) {
					int ele = eleInCache[i];
					if (ele==-1) break;
					// printf("%d ele in cache line, checking %d 's neighbor : \n", count, ele);

                    while (graphMat[getInMap(sortEle, ele)]->next != NULL && flags[graphMat[getInMap(sortEle, ele)]->next -> val]!=0) {
                    	graphMat[getInMap(sortEle, ele)]->next = graphMat[getInMap(sortEle, ele)]->next->next;
                    }
                    // printf("already clean head \n");

					if (graphMat[getInMap(sortEle, ele)]->next != NULL && (graphMat[getInMap(sortEle, ele)]->next -> weight) > localMax) {
						localMax = graphMat[getInMap(sortEle, ele)]->next -> weight ;
						e1 = ele;
						e2 = graphMat[getInMap(sortEle, ele)]->next -> val;
						// printf("bigger! %d  %d", e1, e2);

					}
				}
				if (localMax ==0 ) break;
				// printf("found local max %d to %d \n", e1, e2);
			}
			else {
				// printf("cache line full !\n");
				for (int i =0; i<cacheSize; i++) {
					int ele = eleInCache[i];
					if (ele ==-1) printf("cacheline is not full! \n");
					else graphMat[getInMap(sortEle, ele)]->next = NULL;
				}

			}
		}
    }

    if (index<M) {
		for (int i=0; i<M ; i++) {
			if (flags[i]==0) mapping[index++] = i;
		}
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	//printf("The total generate mapping time is %lu sec \n" ,  (end.tv_sec - start.tv_sec) );
}



void graphPacking( MatrixInfo * mat, MatrixInfo * vec, int * mapX, int * mapY , int warpNum) {
	//printf("start packing...\n");
	struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    Map * sortEleX = createMap(BIG_PRIME);
    Map * sortEleY = createMap(BIG_PRIME);

    int * sizeX = (int*)malloc(sizeof(int));
    int * sizeY = (int*)malloc(sizeof(int));


    LinkedNode ** graphMatX = generateGraph(mat, warpNum, sortEleX, sizeX, 1);
    LinkedNode ** graphMatY = generateGraph(mat, warpNum, sortEleY, sizeY, 0);


    // printf("printing graph X ...\n");
    // for (int i=0; i<mat->N; i++) {
    //     LinkedNode * tmp = graphMatX[i];
    //     while( tmp != NULL ) {
    // 	    printf( "val and weight : %d  %d\n", tmp->val ,tmp -> weight);
    // 	    tmp = tmp->next;
    //     }
    // }


	generateMapping(graphMatX, mat->N, * sizeX, mapX, CACHE_LINE_SIZE, sortEleX);
	generateMapping(graphMatY, mat->M, * sizeY, mapY, CACHE_LINE_SIZE, sortEleY);

	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	//printf("The total  packing time is %lu min \n" ,  (end.tv_sec - start.tv_sec)/60 );

}


