#include <stdio.h>
#include <string.h>
#include "spmv.cuh"
#include "genresult.cuh"
#include "mmio.h"

void logError(const char * errArg, const char * eMsg){
	if(eMsg != NULL)
		printf("Error: %s\n", eMsg);
	if(errArg != NULL)
		printf("Error found near: '%s'\n", errArg);
	puts("USAGE: spmv -mat [matrix file] -ivec [vector file] -alg [segment|design] -blksize [blocksize] -blknum [blocknum]");
	puts("Where the order of the parameters and string case do not matter");
	puts("And the algorithms are:");
	puts("     - segment = simple segment based scan approach");
	puts("     - ftp = first touch packing algorithm");
	puts("     - graph = graph based packing algorithm");
}

typedef enum{CMDLN_ARG_NULL, CMDLN_ARG_MAT = 1, CMDLN_ARG_VEC = 2, CMDLN_ARG_ALG = 4, CMDLN_ARG_BLOCK = 8, CMDLN_ARG_BLOCKNUM = 16, CMDLN_ARG_ERR = 32} CmdLnArg;

CmdLnArg getArgType(const char * argv){
	if(strcasecmp(argv, "-mat") == 0)
		return CMDLN_ARG_MAT;
	else if(strcasecmp(argv, "-ivec") == 0)
		return CMDLN_ARG_VEC;
	else if(strcasecmp(argv, "-alg") == 0)
		return CMDLN_ARG_ALG;
	else if(strcasecmp(argv, "-blksize") == 0)
		return CMDLN_ARG_BLOCK;
	else if(strcasecmp(argv, "-blknum") == 0)
		return CMDLN_ARG_BLOCKNUM;
	else
		return CMDLN_ARG_ERR;
}

typedef enum {ALG_SEGMENT = 0, ALG_DESIGN = 1, ALG_FTP = 2, ALG_GRAPH = 3} AlgType;

int populateAlgType(const char * argv, AlgType * toPop){
	if(strcasecmp(argv, "segment") == 0){
		*toPop = ALG_SEGMENT;
		return 1;
	}else if(strcasecmp(argv, "design") == 0){
		*toPop = ALG_DESIGN;
		return 1;
	}else if(strcasecmp(argv, "ftp") == 0){
		*toPop = ALG_FTP;
		return 1;
	}else if(strcasecmp(argv, "graph") == 0){
		*toPop = ALG_GRAPH;
		return 1;
	}else {
		return 0;
	}
}

int doSpmv(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, AlgType how, int blockSize, int blockNum){
	switch(how){
		case ALG_SEGMENT:
			//for(int i=1;i<=64;i++){
			//	getMulScan(mat, vec, res, 256, i);
			//}
			getMulScan(mat, vec, res, blockSize, blockNum);
			return 1;
		case ALG_DESIGN:
			getMulDesign(mat, vec, res, blockSize, blockNum);
			return 1;
		case ALG_FTP:
			getMulReorder(mat, vec, res, blockSize, blockNum, 0);
			return 1;
		case ALG_GRAPH:
			getMulReorder(mat, vec, res, blockSize, blockNum, 1);
			return 1;
		default:
			return 0;
	}
}

/* a function that verifies the output with the provided sample solution
 * the function will print the total number of incorrect rows */
int verify(const int nz, const int M, const int *rIndex, const int *cIndex, const float *val, const float *vec, const float *res) {

	float *correct = (float*)malloc(sizeof(float) * M);
	memset(correct, 0, sizeof(float) * M);

	/* get the correct output vector */

	for (int i = 0; i < nz; ++i) {
		correct[rIndex[i]] += val[i] * vec[cIndex[i]];
	}

	int o = 0; // the total number of incorrect rows, initialized to 0

	for (int i = 0; i < M; ++i) {
		float l = correct[i] > 0 ? correct[i] : -1*correct[i];
		float m = res[i] > 0 ? res[i] : -1*res[i];
		float k = l - m > 0 ? l - m : m - l;
		float rel = k / l;
		if (rel > .01) {
			o++;
			//printf("Yours - %e, correct - %e, row - %d, Relative error - %f\n", res[i], correct[i], i, rel);
		}
	}

	return o;
}

void printMatrix(MatrixInfo * matrix){
	//printf("wtffffffffffffffffffff");
	int t = 4, nz = matrix->nz;
	for(int i=0;i<t;i++){
		printf("The matrix FIRST %d elements: ", t);
		printf("%d %d %f\n", matrix->rIndex[i], matrix->cIndex[i], matrix->val[i]);
	}
	for(int i=0;i<t;i++){
		printf("The matrix LAST %d elements: ", t);
		printf("%d %d %f\n", matrix->rIndex[nz-t+i], matrix->cIndex[nz-t+i], matrix->val[nz-t+i]);
	}
	printf("The matrix ends.:\n");
}

int main(int argc, char ** argv){
	if(argc != 11){
		logError(NULL, NULL);
		return 1;
	}

	//This is so that the arguments can be presented in any order with the blocksize defaulting to 1024
	int cumArgs = CMDLN_ARG_NULL;
	CmdLnArg argOrder[5];
	int i;
	for(i = 1; i < argc; i += 2){
		CmdLnArg currArg = getArgType(argv[i]);
		if(currArg == CMDLN_ARG_ERR || currArg & cumArgs){
			logError(argv[i], "Invalid or duplicate argument.");
			return 1;
		}else{
			argOrder[i/2] = currArg; //May the truncation be ever in our favor.
			cumArgs |= currArg;
		}
	}

	if(! (31 & cumArgs)){
		logError(NULL, "Missing arguments!");
		return 1;
	}

	char * mFile, * vFile;
	AlgType algo; //Si, debe ser algo!
	int blockSize;
	int blockNum;

	for(i = 0; i < (argc - 1)/2; i++){
		switch(argOrder[i]){
			case CMDLN_ARG_ALG:
				if(!populateAlgType(argv[i * 2 + 2], &algo)){
					logError(argv[i * 2 + 2], "Unsupported algorithm");
					return 1;
				}
				break;
			case CMDLN_ARG_MAT:
				mFile = argv[i * 2 + 2];
				break;
			case CMDLN_ARG_VEC:
				vFile = argv[i * 2 + 2];
				break;
			case CMDLN_ARG_BLOCK:
				if(sscanf(argv[i * 2 + 2], "%d", &blockSize) != 1 || blockSize <= 0){
					logError(argv[i * 2 + 2], "Block size must be a positive integer (greater than 0)");
					return 1;
				}
				break;
			case CMDLN_ARG_BLOCKNUM:
				if(sscanf(argv[i * 2 + 2], "%d", &blockNum) != 1 || blockNum <= 0){
					logError(argv[i * 2 + 2], "Block num must be a positive integer (greater than 0)");
					return 1;
				}
				break;

			default:
				puts("Logic is literally broken. This should never be seen!");
		}
	}

	switch(algo){
	case ALG_SEGMENT:
		//for(int i=1;i<=64;i++){
		//	getMulScan(mat, vec, res, 256, i);
		//}
		printf("\n### Algorithm: segment scan (without improvement) ###\n\n");
		break;
	case ALG_DESIGN:
		printf("\n### Algorithm: segment scan (without improvement) ###\n\n");
		break;
	case ALG_FTP:
		printf("\n### Algorithm: segment scan (First Touch Packing) ###\n\n");
		break;
	case ALG_GRAPH:
		printf("\n### Algorithm: segment scan (Graph Based Packing) ###\n\n");
		break;
	default:
		return 0;
	}

	printf("Reading matrix from %s\n", mFile);
	MatrixInfo * matrix = read_file(mFile);
	if(matrix == NULL){
		logError(mFile, "Error regarding matrix file.");
		return 1;
	}

	printf("Reading vector from %s\n\n", vFile);
	MatrixInfo * vector = read_vector_file(vFile, matrix->N);
	if(vector == NULL){
		logError(mFile, "Error regarding vector file.");
		return 1;
	}

	MatrixInfo * product = initMatrixResult(matrix->M, blockSize);
	cudaError_t err;
	if(!doSpmv(matrix, vector, product, algo, blockSize, blockNum)
			|| (err = cudaDeviceSynchronize()) != cudaSuccess
			|| !writeVect(product, "output.txt")){

		printf("\x1b[31m%s\x1b[0m\n", cudaGetErrorString(err));
		logError(NULL, "Failed to produce output");
	} else {
		printf("Verifying...");
		int o = verify(matrix->nz, matrix->M, matrix->rIndex, matrix->cIndex, matrix->val, vector->val, product->val);
		printf("\t %d Error rows out of %d found \n", o, matrix->M);
		float correctRate = 1 - o * 1.0 / matrix->M;
		printf("\t %f%% Correct rate \n", correctRate*100);
		puts(correctRate > 1-0.001?"Test passed!":"Test failed!");
		/*printf("\nprinting result...\n");
		for(int i=0;i<product->nz;i++){
			printf("%f, ", product->val[i]);
		}*/

		freeMatrixInfo(matrix);
		freeMatrixInfo(vector);
       	freeMatrixInfo(product);
		return 1;
	}

	freeMatrixInfo(matrix);
	freeMatrixInfo(vector);
	freeMatrixInfo(product);

	return 0;
}
