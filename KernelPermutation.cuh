#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void applyPermutation(char* text, char* permutatedText, int* permutation, int permutationLength,int textLength);
__global__ void inversePermutation(int* permutation, int* inverse_permutation, const int permutationLength);