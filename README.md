# Parallel Permutation Cipher
 
In permutation ciphering, both in C++ and CUDA, we have a method that permutes a text based on a given permutation. This method is used for both encryption and decryption. In the sequential version, we iterate over each character and permute it, whereas in the parallel version, the i-th thread permutes the i-th character of the text. To demonstrate how the algorithm works, let's take an example.

We need to permute the following text using the given permutation {2,1,0}. When permuting the letter 'F', we take the index 5 and map it to the range [1,2], resulting in the index 2. According to the given permutation, π(2) = 0. Now, we find the first index of the block to which index 5 belongs (i.e., the block [3,5]), and we add the value obtained, π(2) = 0, to this index to get the final index 3.


## Kernel
```
__global__ void applyPermutation(char* text, char* permutatedText, int* permutation, int permutationLength, int textLength) {
	int i = threadIdx.x+blockIdx.x*blockDim.x;
	if (i < textLength) {
		int p = permutation[i % permutationLength];
		int j = (i / permutationLength) * permutationLength + p;
		permutatedText[j] = text[i];
	}
}
```


## Padding 

In permutation ciphering, padding is considered, which is necessary when the length of the text is not a multiple of the key length. In this case, a symbol is added to represent padding, for example, the symbol @. If we have the text "ABCDEFG," padding would occur as follows:

```
string permutationEncrypt(string text, int* permutation, int permutationLength) {
 	int m = text.length() % permutationLength;
 	//Add the padd symbol if padding is needed.
 	if (m != 0) {
  		int padding = permutationLength - m;
  		for (int i = 0; i < padding; i++) {
 			  text += paddSymbol;
 	 	}
 	}
 	string encryptedText = applyPermutation(text, permutation,permutationLength);
 	return encryptedText;
}
```


When decrypting the text "ABCDEFG@@", we look at the last block. If we encounter any @ (padding symbol), we remove every character after the padding symbol (which means we remove all padding symbols) and obtain the text "ABCDEFG".

```
string permutationDecrypt(string text, int* permutation, int permutationLength) {
	int* inv_permutation = getInversePermutation(permutation, permutationLength);
	string decryptedText = applyPermutation(text, inv_permutation, permutationLength);
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
```




