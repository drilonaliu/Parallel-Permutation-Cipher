#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
using namespace std;

string parallelPermutationEncrypt(string text, int* permutation, int permutationLength);
string parallelPermutationDecrypt(string text, int* permutation, int permutationLength);
int* getInversePermutation(int* permutation, const int permutationLength);
string parallelApplyPermutation(string text, int* permutation, int permutationLength);

