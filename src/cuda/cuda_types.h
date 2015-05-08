/*
 * cuda_types.h
 *
 *  Created on: Mar 24, 2015
 *      Author: Fernando B Oliveira - fboliveira25@gmail.com
 *
 *  Description:
 *	
 */

#ifndef CUDA_TYPES_H_
#define CUDA_TYPES_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// http://choorucode.com/2011/04/09/thrust-passing-device_vector-to-kernel/
// Template structure to pass to kernel
template < typename T >
struct KernelArray
{
    T*  _array;
    int _size;
};

// Function to convert device_vector to structure
template < typename T >
KernelArray< T > convertToKernel( thrust::device_vector< T >& dVec )
{
    KernelArray< T > kArray;
    kArray._array = thrust::raw_pointer_cast( &dVec[0] );
    kArray._size  = ( int ) dVec.size();

    return kArray;
}


#endif /* CUDA_TYPES_H_ */
