/*
 * cuda_kernels.h
 *
 *  Created on: Mar 24, 2015
 *      Author: Fernando B Oliveira - fboliveira25@gmail.com
 *
 *  Description:
 *	
 */

#ifndef CUDA_KERNELS_H_
#define CUDA_KERNELS_H_

#include <cuda_runtime.h>
#include "cuda_types.h"

__global__ void mutateIndividual(KernelArray< int > genes,
		KernelArray< int > keys, KernelArray< int > res);

#endif /* CUDA_KERNELS_H_ */
