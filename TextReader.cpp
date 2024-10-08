#include <iostream>
#include <fstream>
#include <sstream> 
#include  "TextReader.h"
using namespace std;

extern string readTextFile(string fileName);
extern void outputTextFile(string fileName, string content);

//Returns as a string the txt content
string readTextFile(string fileName)
{
    ifstream inFile;
    inFile.open(fileName); // open the input file
    stringstream strStream;
    strStream << inFile.rdbuf();  // read the file
    string str = strStream.str(); // str holds the content of the file
    return str;
}

//Writes as output a text file with a content
void outputTextFile(string fileName, string content)
{
    ofstream out(fileName);
    out << content;
    out.close();
}
