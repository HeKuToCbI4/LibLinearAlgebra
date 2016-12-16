#include "Matrix.cuh"
#include "Vector.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ComplexNumber.h"

//template __declspec(dllexport) class Matrix<ComplexNumber>;
template <class T>
__global__ void matMulKernel(const T* a, const T* b, T* c, size_t ay, size_t by, size_t cy)
{
	T cval = 0;
	size_t col = blockIdx.x*blockDim.x + threadIdx.x;
	size_t row = blockIdx.y*blockDim.y + threadIdx.y;
	for (size_t e = 0; e < ay; ++e)
	{
		cval += a[row*ay + e] * b[e*by + col];
	}
	c[row*cy + col] = cval;
}

template <class T>
__global__ void matCompareKernel(const T* a, const T* b, bool* res, size_t ay)
{
	size_t col = blockIdx.x*blockDim.x + threadIdx.x;
	size_t row = blockIdx.y*blockDim.y + threadIdx.y;
	if (a[row*ay + col] != b[row*ay + col])
		*res = false;
}

template <class T>
Matrix<T>::Matrix()
{
	matrix = Vector<Vector<T>>();
}

template <class T>
Matrix<T>::Matrix(size_t x, size_t y)
{
	matrix = Vector<Vector<T>>(x);
	for (size_t i(0); i < x; i++)
		matrix[i] = Vector<T>(y);
}

template <class T>
void Matrix<T>::push_back(Vector<T> vec)
{
	matrix.push_back(vec);
}

template <class T>
Matrix<T>::Matrix(const Vector<T>& vec): Matrix()
{
	matrix.push_back(vec);
}

template <class T>
Matrix<T>::Matrix(const Matrix<T>& mat)
{
	for (auto i(0); i < mat.get_x_dim(); i++)
		matrix.push_back(Vector<T>(mat[i]));
}

template <class T>
double Matrix<T>::determinant()
{
	return Determinant(*this);
}

template <class T>
Matrix<T> Matrix<T>::transponate()
{
	Matrix<T> res(get_y_dim(), get_x_dim());
	for (auto i(0); i < get_x_dim(); i++)
		for (auto j(0); j < get_y_dim(); j++)
			res[j][i] = matrix[i][j];
	return res;
}

template <class T>
const Vector<T>& Matrix<T>::operator[](size_t index) const
{
	return matrix[index];
}

template <class T>
Vector<T>& Matrix<T>::operator[](size_t index)
{
	return matrix[index];
}

template <class T>
size_t Matrix<T>::get_x_dim() const
{
	return matrix.size();
}

template <class T>
size_t Matrix<T>::get_y_dim() const
{
	return matrix[0].size();
}
