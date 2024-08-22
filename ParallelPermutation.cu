#include "ParallelPermutation.cuh"
#include "KernelPermutation.cuh"

using namespace std;

/*
If text is not a multiple of key, we add the symbol " ` " until
the last block is a multiple of the text.
When we decrypt, we remove the paddSymbol
*/
char paddSymbol = '`';

/*
* Encrypts in parallel a text using the permutation cipher.
*
* @param text - plain text.
* @param permutation - key.
* @param permutationLength - length of permutation.
*
* @return encrypted text.
*
*/
string parallelPermutationEncrypt(string text, int* permutation, int permutationLength) {
	int m = text.length() % permutationLength;
	//Add the padd symbol if padding is needed.
	if (m != 0) {
		int padding = permutationLength - m;
		for (int i = 0; i < padding; i++) {
			text += paddSymbol;
		}
	}
	string encryptedText = parallelApplyPermutation(text, permutation, permutationLength);
	return encryptedText;
}


/*
* Decrypts in parallel a text using the permutation cipher.
*
* @param text - plain text.
* @param permutation - key.
* @param permutationLength - length of permutation.
*
* @return encrypted text.
*
*/
string parallelPermutationDecrypt(string text, int* permutation, int permutationLength) {
	int* inv_permutation = getInversePermutation(permutation, permutationLength);
	string decryptedText = parallelApplyPermutation(text, inv_permutation, permutationLength);

	/*Check if padding was added by going to the last block.
	If it has the padding symbol, remove everything after it.*/
	for (int i = text.length() - permutationLength - 1; i < text.length(); i++) {
		char letter = decryptedText[i];;
		if (letter == paddSymbol) {
			decryptedText.erase(i);
			break;
		}
	}

	return decryptedText;
}

/*
* Applies any kind of permutation on a given text in parallel.
* Method used in both encryption and decryption.
*
* @return the permutated text
*/
string parallelApplyPermutation(string text, int* permutation, int permutationLength) {

	//String to char array
	char* text_arr = new char[text.length() + 1];
	strcpy(text_arr, text.c_str());
	int size = (text.length() + 1) * sizeof(char);

	char* permutated_arr = new char[text.length() + 1];
	
	//Device pointers
	char* d_text;
	char* d_permutatedText;
	int* d_permutation = 0;


	//Memory allocation 
	cudaMalloc((void**)&d_text, size);
	cudaMalloc((void**)&d_permutatedText, size);
	cudaMalloc((void**)&d_permutation, permutationLength * sizeof(int));

	//Memory copy
	cudaMemcpy(d_text, text_arr, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_permutation, permutation, permutationLength * sizeof(int), cudaMemcpyHostToDevice);

	//Launch Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (int)ceil(text.length() * 1.0 / threadsPerBlock);
	int totalThreadsLaunched = threadsPerBlock * blocksPerGrid;
	applyPermutation <<< blocksPerGrid, threadsPerBlock >> > (d_text, d_permutatedText, d_permutation, permutationLength, text.length());

	//Wait for gpu to excecute all threads
	cudaDeviceSynchronize();

	//Retrieve Results
	cudaMemcpy(permutated_arr, d_permutatedText, size, cudaMemcpyDeviceToHost);

	string permutated = permutated_arr;

	//Free memory
	cudaFree(d_text);
	cudaFree(d_permutation);

	delete[] text_arr;

	return permutated;
}

/*
* Calculates the inverse permutation for a given one.
*
* @return inversed permutation.
*/
int* getInversePermutation(int* permutation, const int permutationLength)
{
	int* inv_permutation = new int[permutationLength];
	for (int i = 0; i < permutationLength; i++) {
		inv_permutation[permutation[i]] = i;
	}
	return inv_permutation;
}