#include <iostream>
#include "ParallelPermutation.cuh";
#include "TextReader.h"


int main() {
	
	const int permutationLength = 8;
	int permutation[permutationLength] = { 4, 5, 6, 3, 2, 0, 1,7 };

	// Encryption and decryption of a string
	string text = "Parallel Programming";
	string encrypted = parallelPermutationEncrypt(text, permutation, permutationLength);
	string decrypted = parallelPermutationDecrypt(encrypted, permutation, permutationLength);

	cout << "\nOriginal: " + text
		<< "\nEncrypted: " + encrypted
		<< "\nDecrypted: " + decrypted;

	// Encryption and decryption on a text file
	string plainTxt = readTextFile("Texts/PlainText.txt");
	string encryptedTxt = parallelPermutationEncrypt(plainTxt, permutation, permutationLength);
	outputTextFile("Texts/Encrypted.txt", encryptedTxt);
	string decryptedTxt = parallelPermutationDecrypt(readTextFile("Texts/Encrypted.txt"), permutation, permutationLength);
	outputTextFile("Texts/Decrypted.txt", decryptedTxt);

	return 0;
}