/*
 * cuda_functions.h
 *
 *  Created on: Mar 24, 2015
 *      Author: Fernando B Oliveira - fboliveira25@gmail.com
 *
 *  Description:
 *	
 */

#ifndef CUDA_FUNCTIONS_H_
#define CUDA_FUNCTIONS_H_

#include <vector>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "ManagedMatrix.h"

#include "../classes/Random.hpp"
#include "../classes/Util.hpp"

#include "cuda_kernels.h"
#include "cuda_types.h"

void cudaMutate(::vector<int>& genes);
void cudaTeste();

#endif /* CUDA_FUNCTIONS_H_ */
