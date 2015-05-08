/*
 * cuda_functions.cpp
 *
 *  Created on: Mar 24, 2015
 *      Author: Fernando B Oliveira - fboliveira25@gmail.com
 *
 *  Description:
 *	
 */

#include "cuda_functions.h"

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

struct print
{
  __host__ __device__
  int operator()(int x)
  {
	  printf("%d\n", x);
	  return x;
  }
};

template <typename Iterator>
void print_range(const std::string& name, Iterator first, Iterator last)
{
    typedef typename std::iterator_traits<Iterator>::value_type T;

    std::cout << name << ": ";
    thrust::copy(first, last, std::ostream_iterator<T>(std::cout, " "));
    std::cout << "\n";
}

void cudaMutate(::vector<int>& genes) {

	std::vector<int> key;
	Random::randPermutation(genes.size(), 0, key);
	int s = genes.size();

	//Util::print(genes);

	thrust::host_vector<int> h_res(s);
	thrust::device_vector<int> d_genes = genes;
	thrust::device_vector<int> d_keys = key;
	thrust::device_vector<int> d_res = genes;

	//dim3 threadsPerBlock(32, 32);
	//dim3 numBlocks(s / threadsPerBlock.x, s / threadsPerBlock.y);

	// Invoke kernel
	int threadsPerBlock = 32;
	int blocksPerGrid = (s + threadsPerBlock - 1) / threadsPerBlock;

	mutateIndividual<<<blocksPerGrid, threadsPerBlock>>>( convertToKernel(d_genes), convertToKernel(d_keys),
			convertToKernel(d_res) );
	cudaDeviceSynchronize();

	genes.clear()	;

	// transfer data back to host
	thrust::copy(d_res.begin(), d_res.end(), h_res.begin());
	genes.assign(h_res.begin(), h_res.end());

	//Util::print(genes);

}

void cudaTeste() {
	int data[10] = {0, 1, 3, 6, 0, 2, 5, 4, 7, 0};

	thrust::device_vector<int> d_data(data, data + 10);

	//thrust::transform(thrust::host, d_data.begin(), d_data.end(), d_data, print());

//	for(int i = 0; i < 10; ++i)
//		printf("%d - \t", data[i]);

	ManagedMatrix<int> problem(10, 10);

	problem.set(5,6,-1);
	problem.print();


}



