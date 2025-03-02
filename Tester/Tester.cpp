#include "stdafx.h"
#include <iostream>
#include <exception>
#include <chrono>
#include <string>
#include <fstream>
#include <Windows.h>
#include <sstream>
#include <ComplexNumber.h>
#include <Matrix.cuh>
#include <Vector.cuh>
using namespace std;
using namespace chrono;


template <class T>
void print_vec(Vector<T> vec)
{
	for (auto e : vec)
		cout << e << " ";
	cout << endl;
}

template <class T>
void print_matr(Matrix<T> matr)
{
	for (auto i(0); i < matr.get_x_dim(); i++)
		print_vec(matr[i]);
	getchar();
}



int main()
{

	cout << "PART 1: Testing for Complex number methods:\n";
	cout << "Implementation, constructors testing.\n";
	ComplexNumber a = ComplexNumber();
	double x, y;
	cout << "Input real and imaginary part for testing constructor with parameters.\n";
	cin >> x >> y;
	auto b = ComplexNumber(x, y);
	auto c = ComplexNumber(b);
	cout << "Default complex number: " << a << endl;
	cout << "Complex number with parameters: " << b << endl;
	cout << "Copy of previous number initialized via constructor: " << c << endl;
	cout << "Testing write to file and read from it: ";
	string filename;
	cout << "Input desired filename\n";
	cin >> filename;
	ofstream ofile;
	ifstream ifile;
	ofile.open(filename);
	cout << "Writing to file: " << b << endl;
	ofile << b;
	ofile.close();
	ifile.open(filename);
	b.set_real(10);
	cout << "number was changed to: " << b << endl;
	ifile >> b;
	cout << "Number after reading from file: " << b << endl;
	system("pause");
	ifile.close();
	cout << "Input two numbers in form a+b*i: ";
	cin >> a >> b;
	cout << "Operators test:\n";
	cout << "a+b: " << a + b << endl;
	cout << "a-b: " << a - b << endl;
	cout << "a*b: " << a * b << endl;
	cout << "a/b: " << a / b << endl;
	cout << "unary operators test:\n";
	cout << "a+=b: " << (a += b) << endl;
	cout << "a-=b: " << (a -= b) << endl;
	cout << "a*=b: " << (a *= b) << endl;
	cout << "a/=b: " << (a /= b) << endl;
	cout << "A is now: " << a << endl;
	float num = 2;
	cout << "a+2=" << (a + num) << endl;
	cout << "a-2=" << (a - num) << endl;
	cout << "a*2=" << (a * num) << endl;
	cout << "a/2=" << (a / num) << endl;

	system("pause");
	return 0;
}