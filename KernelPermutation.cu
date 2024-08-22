//for __syncthreads()
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>
#include "KernelPermutation.cuh"

__global__ void applyPermutation(char* text, char* permutatedText, int* permutation, int permutationLength, int textLength) {
	int i = threadIdx.x+blockIdx.x*blockDim.x;
	if (i < textLength) {
		int p = permutation[i % permutationLength];
		int j = (i / permutationLength) * permutationLength + p;
		permutatedText[j] = text[i];
	}
}


__global__ void inversePermutation(int* permutation, int* inv_permutation, const int permutationLength) {
	int i = threadIdx.x;
	inv_permutation[permutation[i]] = i;
}