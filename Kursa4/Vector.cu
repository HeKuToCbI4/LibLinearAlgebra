
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <exception>
#include "ComplexNumber.h"
using namespace std;

template <class T>
__global__ void addKernel(T *c, const T *a, const T *b, size_t N)
{
	size_t i = threadIdx.x*16 + blockIdx.x*blockDim.x*16;
	for (size_t j(0); j < 16; j++)
		if (i + j < N)
			c[i + j] = a[i + j] + b[i + j];
}
template <class T>
__global__ void mulKernel(const T* a, const T* b, T *c, size_t N)
{
	size_t i = threadIdx.x*16 + blockIdx.x*blockDim.x*16;
	for (size_t j(0); j < 16; j++)
		if (i + j < N)
		c[i+j] = a[i+j] * b[i+j];
}

template <class T, class X>
__global__ void mulByNum(const T*a, T* b, const X n, size_t N)
{
	size_t i = threadIdx.x*16 + blockIdx.x*blockDim.x*16;
	for (char j = 0; j < 16; j++)
		if (i + j < N)
			b[i + j] = a[i + j] * n;
}
template <class T>
__global__ void sumVec(const T* a, double* num, size_t size)
{
	num[0] = 0;
	for (size_t i(0); i < size; i++)
		num[0] += a[i];
}




template <class T>
class Vector : public vector<T>
{
public:
	Vector<T>() : vector<T>()
	{
	}
	Vector<T>(size_t size) : vector<T>(size)
	{
	}
	Vector<T>(const Vector<T>& vec)
	{
		for (auto elem : vec)
			this->emplace_back(elem);
	}
	Vector<T> & operator=(const Vector<T>& vec)
	{
		for (auto elem : vec)
			this->emplace_back(elem);
		return *this;
	}
	Vector operator +(const Vector<T>& a)
	{
		Vector<T> b = *this;
		Vector<T> result = Vector(a.size());
		T* d_a;
		T* d_b;
		T* d_c;
		T* h_c;
		if (a.size() != b.size())
			throw exception("YOBA");
		size_t size = sizeof(T)*a.size();
		h_c = (T*)malloc(size);
		if (cudaMalloc(&d_a, size) != cudaSuccess)
		{
			cout << "error in memory allocation\n";
			getchar();
		}
		cudaMalloc(&d_b, size);
		cudaMalloc(&d_c, size);
		if (cudaMemcpy(d_a, &a[0], size, cudaMemcpyHostToDevice) != cudaSuccess)
		{
			cout << "error in memory copy from host to device\n";
			getchar();
		}
		cudaMemcpy(d_b, &b[0], size, cudaMemcpyHostToDevice);
		size_t threadsperblock = 16;
		size_t blockspergrid = (a.size() + threadsperblock * 16 - 1) / threadsperblock / 16;
		addKernel << <blockspergrid, threadsperblock >> > (d_c, d_a, d_b, a.size());
		if (cudaSuccess != cudaGetLastError())
		{
			cout << "Error in kernel!\n";
			getchar();
		}
		if (cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost) != cudaSuccess)
		{
			cout << "error in copying memory from device to host\n";
			getchar();
		}
		result.assign(h_c, h_c + a.size());
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		free(h_c);
		return result;
	}
	friend double operator *(const Vector<T>& a, const Vector<T> &b)
	{
		T* d_a;
		T* d_b;
		T* d_c;
		if (a.size() != b.size())
			throw exception("YOBA");
		size_t size = sizeof(T)*a.size();
		if (cudaMalloc(&d_a, size) != cudaSuccess)
		{
			cout << "error in memory allocation\n";
			getchar();
		}
		cudaMalloc(&d_b, size);
		cudaMalloc(&d_c, size);
		if (cudaMemcpy(d_a, &a[0], size, cudaMemcpyHostToDevice) != cudaSuccess)
		{
			cout << "error in memory copy from host to device\n";
			getchar();
		}
		cudaMemcpy(d_b, &b[0], size, cudaMemcpyHostToDevice);
		size_t threadsperblock = 16;
		size_t blockspergrid = (a.size() + threadsperblock * 16 - 1) / threadsperblock / 16;
		mulKernel <<<blockspergrid, threadsperblock >>> (d_a, d_b, d_c, a.size());
		if (cudaSuccess != cudaGetLastError())
		{
			cout << "Error in kernel!\n";
			getchar();
		}
		double* sum;
		sum = (double*)malloc(sizeof(double));
		double* d_sum;
		cudaMalloc(&d_sum, sizeof(double));
		sumVec<<<1,1>>>(d_c, d_sum, a.size());
		cudaMemcpy(sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		return sum[0];
	}
	template <class X>
	friend Vector operator *(const Vector<T>& a, const X& b)
	{
		Vector<T> result = Vector(a.size());
		T* d_a;
		T* d_b;
		T* h_c;
		size_t size = sizeof(T)*a.size();
		h_c = (T*)malloc( size);
		cudaMalloc(&d_a, size);
		cudaMalloc(&d_b, size);
		cudaMemcpy(d_a, &a[0], size, cudaMemcpyHostToDevice);
		size_t threadsperblock = 16;
		size_t blockspergrid = (a.size() + threadsperblock * 16 - 1) / threadsperblock / 16;
		mulByNum << <blockspergrid, threadsperblock >> > (d_a, d_b, b, a.size());
		cudaMemcpy(h_c, d_b, size, cudaMemcpyDeviceToHost);
		result.assign(h_c, h_c + a.size());
		cudaFree(d_a);
		cudaFree(d_b);
		free(h_c);
		return result;
	}
	template <class X>
	friend Vector operator *(const X& b, const Vector<T>& a)
	{
		return a*b;
	}
	Vector mixed_multiple(const Vector<T>&);
};



template <class T>
Vector<T> Vector<T>::mixed_multiple(const Vector<T>&)
{
}

template <class T>
void print_vec(Vector<T> vec)
{
	for (auto e : vec)
		cout << e << " ";
	cout << endl;
}

template <class T>
class Matrix;

template <class T>
void print_matr(Matrix<T>);

template <class T>
__global__ void matMulKernel(const T* a, const T* b, T* c, size_t ay, size_t by, size_t cy)
{
	T cval=0;
	size_t col = blockIdx.x*blockDim.x + threadIdx.x;
	size_t row = blockIdx.y*blockDim.y + threadIdx.y;
	for (size_t e = 0; e < ay; ++e)
	{
		cval += a[row*ay + e] * b[e*by + col];
	}
	c[row*cy+col]=cval;
}

template<class T>
__global__ void matCompareKernel(const T* a, const T* b, bool* res, size_t ay)
{
	size_t col = blockIdx.x*blockDim.x + threadIdx.x;
	size_t row = blockIdx.y*blockDim.y + threadIdx.y;
	if (a[row*ay + col] != b[row*ay + col])
		*res = false;
}

template <class T>
class Matrix
{
	Vector<Vector<T>> matrix;
public:
	Matrix()
	{
		matrix = Vector<Vector<T>>();
	}
	Matrix(size_t x, size_t y)
	{
		matrix = Vector<Vector<T>>(x);
		for (size_t i(0); i < x; i++)
			matrix[i] = Vector<T>(y);
	}

	void push_back(Vector<T> vec)
	{
		matrix.push_back(vec);
	}

	Matrix(const Vector<T>& vec) : Matrix()
	{
		matrix.push_back(vec);
	}

	Matrix(const Matrix<T>& mat)
	{
		for (auto i(0); i < mat.get_x_dim(); i++)
			matrix.push_back(Vector<T>(mat[i]));
	}
	
	double determinant()
	{
		if (!matrix.size() > 0 || !matrix[0].size > 0)
			throw exception("Matrix is not initialized!");
		if (matrix.size() != matrix[0].size())
			throw exception("Matrix is not square!");
	}

	Matrix<T> transponate()
	{
		Matrix<T> res(get_y_dim(), get_x_dim());
		for (auto i(0); i < get_x_dim(); i++)
			for (auto j(0); j < get_y_dim(); j++)
				res[j][i] = matrix[i][j];
		return res;
	}

	const Vector<T>& operator[](size_t index) const
	{
		return matrix[index];
	}

	Vector<T>& operator[](size_t index)
	{
		return matrix[index];
	}

	size_t get_x_dim() const
	{
		return matrix.size();
	}

	size_t get_y_dim() const
	{
		return matrix[0].size();
	}

	friend Matrix<T> operator + (Matrix<T> &a, Matrix<T> &b)
	{
		if (!((a.get_x_dim() == b.get_x_dim()) && (b.get_y_dim() == a.get_y_dim())))
			throw exception("Matrix sizes are different. Can't add them");\
		Matrix<T> res = Matrix(a.get_x_dim(), a.get_y_dim());
		for (auto i(0); i < a.get_x_dim(); i++)
		{
			res[i] = (a[i] + b[i]);
		}
		return res;
	}
	friend
	Matrix<T> operator*(const Matrix<T> &a, const Matrix<T> &b)
	{
		if (a.get_y_dim() != b.get_x_dim())
			throw exception("Matrices are not fucking multiplable");
		Matrix<T> res(a.get_x_dim(), b.get_y_dim());
		T* h_a;
		T* h_b;
		T* h_c;
		T *d_a, *d_b, *d_c;
		h_a = (T*)(malloc(sizeof(T)*a.get_x_dim()*a.get_y_dim()));
		h_b = (T*)(malloc(sizeof(T)*b.get_x_dim()*b.get_y_dim()));
		h_c = (T*)(malloc(sizeof(T)*a.get_x_dim()*b.get_y_dim()));
		for (size_t i(0); i < a.get_x_dim(); i++)
		{
			for (size_t j(0); j < a.get_y_dim(); j++)
			{
				h_a[i*a.get_y_dim() + j] = a[i][j];
			}
		}
		for (size_t i(0); i < b.get_x_dim(); i++)
		{
			for (size_t j(0); j < b.get_y_dim(); j++)
			{
				h_b[i*b.get_y_dim() + j] = b[i][j];
			}
		}
		cudaMalloc(&d_a, sizeof(T)*a.get_x_dim()*a.get_y_dim());
		cudaMalloc(&d_b, sizeof(T)*b.get_x_dim()*b.get_y_dim());
		cudaMalloc(&d_c, sizeof(T)*a.get_x_dim()*b.get_y_dim());
		cudaMemcpy(d_a, h_a, sizeof(T)*a.get_x_dim()*a.get_y_dim(), cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, h_b, sizeof(T)*b.get_x_dim()*b.get_y_dim(), cudaMemcpyHostToDevice);
		dim3 block(1, 1);
		dim3 grid(b.get_y_dim(), a.get_x_dim());
		matMulKernel << <grid, block >> > (d_a, d_b, d_c, a.get_y_dim(), b.get_y_dim(), res.get_y_dim());
		cudaMemcpy(h_c, d_c, sizeof(T)*a.get_x_dim()*b.get_y_dim(), cudaMemcpyDeviceToHost);
		for (size_t i(0); i < res.get_x_dim(); i++)
			for (size_t j(0); j < res.get_y_dim(); j++)
				res[i][j] = h_c[i*res.get_y_dim() + j];
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		free(h_a);
		free(h_b);
		free(h_c);
		return res;

	}

	friend Matrix<T> operator*(const Matrix<T>&a, const Vector<T> &b)
	{
		Matrix<T> tmp(b);

		return a*tmp;
	}

	friend Matrix<T> operator*(const Vector<T>& b, const Matrix<T>&a)
	{
		Matrix<T> tmp(b);

		return tmp*a;
	}
	friend
	bool operator==(const Matrix<T> a, const Matrix<T>& b)
	{
		if (a.get_x_dim() != b.get_x_dim() || a.get_y_dim() != b.get_y_dim())
			return false;
		bool* res = (bool*)malloc(sizeof(bool));
		*res = true;
		T* h_a;
		T* h_b;
		T *d_a, *d_b;
		bool* d_res;
		
		h_a = (T*)(malloc(sizeof(T)*a.get_x_dim()*a.get_y_dim()));
		h_b = (T*)(malloc(sizeof(T)*b.get_x_dim()*b.get_y_dim()));
		for (size_t i(0); i < a.get_x_dim(); i++)
		{
			for (size_t j(0); j < a.get_y_dim(); j++)
			{
				h_a[i*a.get_y_dim() + j] = a[i][j];
			}
		}
		for (size_t i(0); i < b.get_x_dim(); i++)
		{
			for (size_t j(0); j < b.get_y_dim(); j++)
			{
				h_b[i*b.get_y_dim() + j] = b[i][j];
			}
		}
		cudaMalloc(&d_a, sizeof(T)*a.get_x_dim()*a.get_y_dim());
		cudaMalloc(&d_b, sizeof(T)*b.get_x_dim()*b.get_y_dim());
		cudaMalloc(&d_res, sizeof(bool));
		cudaMemcpy(d_res, res, sizeof(bool), cudaMemcpyHostToDevice);
		cudaMemcpy(d_a, h_a, sizeof(T)*a.get_x_dim()*a.get_y_dim(), cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, h_b, sizeof(T)*b.get_x_dim()*b.get_y_dim(), cudaMemcpyHostToDevice);
		dim3 block(1, 1);
		dim3 grid(b.get_x_dim(), a.get_y_dim());
		matCompareKernel << <grid, block >> >(d_a, d_b, d_res, a.get_y_dim());
		cudaMemcpy(res, d_res, sizeof(bool), cudaMemcpyDeviceToHost);
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_res);
		free(h_a);
		free(h_b);
		bool val = *res;
		free(res);
		return val;
	}
};



template <class T>
void print_matr(Matrix<T> matr)
{
	for (auto i(0); i < matr.get_x_dim(); i++)
		print_vec(matr[i]);
	getchar();
}
int main()
{
	/*ComplexNumber num1(1, 1), num2(2, -3);
	std::cout << num1 << "   " << num2 << endl;
	cin >> num1 >> num2;
	
	cout << "sum " << num1 + num2 << endl;
	cout << "diff " << num1 - num2 << endl;
	cout << "multiple " << num1*num2 << endl;
	cout << "division " << num1 / num2 << endl;
	cout << "conjugation " << num1.get_conjugation() << endl;
	cout << "module " << num1.module() << endl;
	cout << "argument " << num1.argument() << endl;
	getchar();*/
	Vector<int> vec1, vec2, vec3;
	for (int i = 0; i < 15; i++)
	{
		vec1.emplace_back(1*i);
		vec2.emplace_back(2*i);
	}
	Matrix<int> matr;
	Matrix<int> matr3;
	Vector<int> test;
	test.emplace_back(1);
	for (auto i = 0; i < 10; i++)
	{
		matr.push_back(vec1);
		matr3.push_back(vec2);
	} // <3 kek
	print_matr(matr);
	print_matr(matr3.transponate());
	if (matr == matr3)
		cout << "COMPARATION OF MATRICES: TRUE\n";
	else
		cout << "DIS IS GODDAMN FALSE BEAAAACH!\n";
		
	Matrix<int> matr2 = matr*matr3.transponate();
	print_matr(matr2);
	
	vec3 = vec1 + vec1;
	double scalar_multiple = vec1*vec2;
	print_vec(vec3);
	cout << "SCALAR: " << scalar_multiple << endl;
	getchar();
	return 0;
}

