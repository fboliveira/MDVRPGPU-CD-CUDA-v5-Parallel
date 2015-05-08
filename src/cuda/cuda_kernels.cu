/*
 * evolve.cu
 *
 *  Created on: Mar 24, 2015
 *      Author: Fernando B Oliveira - fboliveira25@gmail.com
 *
 *  Description:
 *	
 */

#include "cuda_kernels.h"

__global__ void mutateIndividual(KernelArray< int > genes,
		KernelArray< int > keys, KernelArray< int > res) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	//printf("Thread ID = %d\n", idx);

	if (idx < genes._size) {
		//printf(" %d -> %d\n", idx, keys._array[ idx ]);
		res._array [ idx ] = genes._array[ keys._array[ idx ] ];
		//printf("res[%d] = %d\n", idx, res._array [ idx ]);
	}

}
