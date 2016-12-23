#include <iostream>
#include <chrono>
#include <string>
#include <fstream>
#include "ComplexNumber.cuh"
#include "Matrix.cuh"
#include "Vector.cuh"
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
	ofstream output_of_program;
	output_of_program.open("Simple output.txt");
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
	cout << "ENOUGH COMPLEX NUMBERS 4 YA." << endl;
	cout << "It's hiiiiiiiigh noooon. Let's do something with vectors.\n Type in length of testing vector 4 example." << endl;
	int len;
	cin >> len;
	Vector<ComplexNumber> vec1(len), vec2(len), vec3(len);
	for (int i = 0; i<len; i++)
	{
		vec1[i] = ComplexNumber(i + 1, i*i);
		vec2[i] = ComplexNumber(2*i*i+1.5, 3);
		vec3[i] = ComplexNumber(3, 6-i);
	}
	cout << "Vectors now look like dis: " << endl;
	print_vec(vec1);
	print_vec(vec2);
	print_vec(vec3);
	cout << "Now let's see does CUDA functions work properly:\n";
	cout << "SUM OF VECTORS: " << endl;
	print_vec(vec1 + vec2);
	cout << "Difference of vectors\n";
	print_vec(vec1 - vec2);
	cout << "LALALA IT'S TIME TO SEE IF SCALAR MULTIPLE IS REAL: " << (vec1*vec2) << endl;
	try {
		cout << "And indeed we want to see if we can calculate sth like mixed multiple vec1 and vec2: " << mixed_multiple(vec1, vec2, vec3) << endl;
	}
	catch(exception e)
	{
		cout << e.what() << endl;
	}
	cout << "Result of multiplication vec1 by 2 and 3 by vec1" << endl;
	print_vec(vec1 * 2);
	print_vec(3 * vec1);
	cout << "I CAN COMPARE VECTORS. RES OF COMPARING 1 and 3 and 1 and 1: " << (vec1 == vec3) << "  " << (vec1 == vec1) << endl;
	cout << "I CAN READ AND WRITE THEM TO FILES AND BUFFERED STREAMS: " << vec1 << endl;
	cout << "first - count. then elems.\n";
	cin >> vec1;
	cout << "Binary += and -= with vec1 and vec2" << endl;
	cout << (vec1 += vec2) << endl;
	cout << (vec1 -= vec2) << endl;
	cout << "DATS ALL WITH VECTORS FOR NOW." << endl;
	cout << "WAIT FOR IT. LET'S TRY COMPLEX NUMBER VECTORS!1!1" << endl;
	cout << "MATRIX TIME DUUUUUUUDE" << endl;
	int n, m;
	cout << "Test for matrices. Input x and y" << endl;
	cin >> n >> m;
	Matrix<ComplexNumber> am(n, m), bm(n, m), cm, dm;
	Vector<ComplexNumber> vec(n);
	cout << "Filling a with ComplexNumber(i, j), filling b with i+j" << endl;
	for (auto i(0); i < n; i++)
		for (auto j(0); j < m; j++)
		{
			am[i][j] = ComplexNumber(i, j);
			bm[i][j] = i + j;
			vec[i] = ComplexNumber(i + 1, j*j);
		}
	cout << "Matrices" << endl;
	print_matr(am);
	print_matr(bm);
	cout << "Operators test" << endl;
	cout << "a+b" << endl;
	print_matr(am + bm);
	cout << "a-b" << endl;
	print_matr(am - bm);
	cout << "Matrix a == b and a!=b: " << (am==bm) << "  " << (am!=bm) << endl;
	cout << "Multiple matrix by number a*2 and 3*b" << endl;
	print_matr(am * 2);
	print_matr(3 * bm);
	cout << "Matrix a determinant: " << am.determinant() << endl;
	cout << "Multiple matrix by vector of size n (vec*matr)" << endl;
	print_vec(vec);
	print_matr(vec*am);
	cout << "Thats all for now!" << endl << "BruteForce part incoming!" << endl;
	system("pause");
	output_of_program.close();
	return 0;
}