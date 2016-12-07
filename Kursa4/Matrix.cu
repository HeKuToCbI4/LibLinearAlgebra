
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <exception>
#include "Vector.cu"
using namespace std;
template <class T>
__global__ void sum_kernel(const T** &matrix1, const T** & matrix2, T** matrix_res, size_t x_dim, size_t y_dim)
{
	size_t x = threadIdx.x * 16 + blockIdx.x*blockDim.x * 16;
	size_t y = threadIdx.y * 16 + blockIdx.y*blockDim.y * 16;
	for (size_t j(0); j < 16; j++)
		for (size_t i(0); i < 16; i++)
			if (i + x < x_dim && j + y < y_dim)
				matrix_res = matrix[i + x][j + y];
}

template <class T>
class Matrix
{
private:
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
	double determinant()
	{
		if (!matrix.size() > 0 || !matrix[0].size > 0)
			throw exception("Matrix is not initialized!");
		if (matrix.size() != matrix[0].size())
			throw exception("Matrix is not square!");
	}
	const Vector<T> operator[](size_t index) const
	{
		return matrix[index];
	}
	Vector<T> operator[](size_t index)
	{
		return marix[index];
	}
	friend Matrix<T> operator + (const Matrix<T> &a, const Matrix<T> &b)
	{
		T** d_a;
		T** d_b;
		T** d_res;
		T** h_res;
		T** h_a = &a[0][0];
		T** h_b = &b[0][0];
		h_res[i] = (T*)malloc(a[0].size()*sizeof(T));

	}

};