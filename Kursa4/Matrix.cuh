#pragma once
#include "cuda_runtime.h"
#include <iostream>
#include "Protector.cuh"
#include "device_launch_parameters.h"
#include <exception>
using namespace std;

template <class T>
class Vector;


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
template<class T, class X>
__global__ void mulMatrByNum(const T* a, const X num, T* c, size_t ax, size_t ay)
{
	size_t row = blockIdx.x*blockDim.x + threadIdx.x;
	size_t col = blockIdx.y*blockDim.y + threadIdx.y;
	if (row < ax && col < ay)
		c[row*ay + col] = a[row*ay + col] * num;
}

template<class T>
__global__ void matrAddKernel(const T* a, const T* b, T* c, size_t ax, size_t ay)
{
	size_t row = blockIdx.x*blockDim.x + threadIdx.x;
	size_t col = blockIdx.y*blockDim.y + threadIdx.y;
	if (row < ax && col < ay)
		c[row*ay + col] = a[row*ay + col] + b[row*ay + col];
}

template<class T>
__global__ void matrDiffKernel(const T* a, const T* b, T* c, size_t ax, size_t ay)
{
	size_t row = blockIdx.x*blockDim.x + threadIdx.x;
	size_t col = blockIdx.y*blockDim.y + threadIdx.y;
	if (row < ax && col < ay)
		c[row*ay + col] = a[row*ay + col] - b[row*ay + col];
}

template <class T>
class Matrix
{
	Vector<Vector<T>> matrix;
	Protector* protector = Protector::get_instance();
public:
	Matrix();

	Matrix(size_t x, size_t y);

	void push_back(const Vector<T>& vec);

	Matrix(const Vector<T>& vec);

	Matrix(const Matrix<T>& mat);

	T determinant();

	friend
		T Determinant(Matrix<T> matr)
	{
		if (!matr.get_x_dim() > 0 || !matr.get_y_dim() > 0)
			throw std::exception("Matrix is not initialized!");
		if (matr.get_x_dim() != matr.get_y_dim())
			throw exception("Matrix is not square!");
		int i, j, j1, j2;
		T det;
		Matrix<T> m;
		if (matr.get_x_dim() == 1)
			return matr[0][0];
		if (matr.get_x_dim() == 2) {
			det = matr[0][0] * matr[1][1] - matr[1][0] * matr[0][1];
		}
		else {
			det = 0;
			m = Matrix(matr.get_x_dim() - 1, matr.get_x_dim() - 1);
			for (j1 = 0; j1 < matr.get_x_dim(); j1++)
			{

				for (i = 1; i < matr.get_x_dim(); i++)
				{
					j2 = 0;
					for (j = 0; j < matr.get_x_dim(); j++)
					{
						if (j == j1)
							continue;
						m[i - 1][j2] = matr[i][j];
						j2++;
					}
				}
				det += T(pow(-1.0, 1.0 + j1 + 1.0)) * matr[0][j1] * Determinant(m);
			}
		}
		return(det);

	}

	Matrix<T> transponate();

	const Vector<T>& operator[](size_t index) const;

	Vector<T>& operator[](size_t index);

	size_t get_x_dim() const;

	size_t get_y_dim() const;

	friend
		Matrix<T> operator+(Matrix<T> &a, Matrix<T> &b)
	{
		if (!((a.get_x_dim() == b.get_x_dim()) && (b.get_y_dim() == a.get_y_dim())))
			throw exception("Matrix sizes are different. Can't add them"); \
			Matrix<T> res = Matrix(a.get_x_dim(), a.get_y_dim());
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
		cudaMalloc(&d_c, sizeof(T)*a.get_x_dim()*a.get_y_dim());
		cudaMemcpy(d_a, h_a, sizeof(T)*a.get_x_dim()*a.get_y_dim(), cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, h_b, sizeof(T)*b.get_x_dim()*b.get_y_dim(), cudaMemcpyHostToDevice);
		dim3 block(16, 16);
		dim3 grid(a.get_x_dim() / block.x + 1, a.get_y_dim() / block.y + 1);
		matrAddKernel << <grid, block >> > (d_a, d_b, d_c, a.get_x_dim(), a.get_y_dim());
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

	friend
		Matrix<T> operator-(Matrix<T> &a, Matrix<T> &b)
	{
		if (!((a.get_x_dim() == b.get_x_dim()) && (b.get_y_dim() == a.get_y_dim())))
			throw exception("Matrix sizes are different. Can't add them"); \
			Matrix<T> res = Matrix(a.get_x_dim(), a.get_y_dim());
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
		cudaMalloc(&d_c, sizeof(T)*a.get_x_dim()*a.get_y_dim());
		cudaMemcpy(d_a, h_a, sizeof(T)*a.get_x_dim()*a.get_y_dim(), cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, h_b, sizeof(T)*b.get_x_dim()*b.get_y_dim(), cudaMemcpyHostToDevice);
		dim3 block(16, 16);
		dim3 grid(a.get_x_dim() / block.x + 1, a.get_y_dim() / block.y + 1);
		matrDiffKernel << <grid, block >> > (d_a, d_b, d_c, a.get_x_dim(), a.get_y_dim());
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
		dim3 block(16, 16);
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

	friend
		Matrix<T> operator*(const Matrix<T>&a, const Vector<T> &b)
	{
		Matrix<T> tmp(b);

		return a*tmp;
	}

	friend
		Matrix<T> operator*(const Vector<T>& b, const Matrix<T>&a)
	{
		Matrix<T> tmp(b);

		return tmp*a;
	}
	friend
		bool operator==(const Matrix<T> a, const Matrix<T>& b)
	{
		if (a.get_x_dim() != b.get_x_dim() || a.get_y_dim() != b.get_y_dim())
			return false;
		auto res = static_cast<bool*>(malloc(sizeof(bool)));
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
		matCompareKernel << <grid, block >> > (d_a, d_b, d_res, a.get_y_dim());
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


	template<class X>
	friend Matrix<T> operator*(const Matrix<T>& a, const X& num)
	{
		Matrix<T> res = Matrix(a.get_x_dim(), a.get_y_dim());
		T* h_a;
		T* h_c;
		T *d_a, *d_c;
		h_a = (T*)(malloc(sizeof(T)*a.get_x_dim()*a.get_y_dim()));
		h_c = (T*)(malloc(sizeof(T)*a.get_x_dim()*a.get_y_dim()));
		for (size_t i(0); i < a.get_x_dim(); i++)
		{
			for (size_t j(0); j < a.get_y_dim(); j++)
			{
				h_a[i*a.get_y_dim() + j] = a[i][j];
			}
		}
		cudaMalloc(&d_a, sizeof(T)*a.get_x_dim()*a.get_y_dim());
		cudaMalloc(&d_c, sizeof(T)*a.get_x_dim()*a.get_y_dim());
		cudaMemcpy(d_a, h_a, sizeof(T)*a.get_x_dim()*a.get_y_dim(), cudaMemcpyHostToDevice);
		dim3 block(16, 16);
		dim3 grid(a.get_x_dim() / block.x + 1, a.get_y_dim() / block.y + 1);
		mulMatrByNum << <grid, block >> > (d_a, num, d_c, a.get_x_dim(), a.get_y_dim());
		cudaMemcpy(h_c, d_c, sizeof(T)*a.get_x_dim()*a.get_y_dim(), cudaMemcpyDeviceToHost);
		for (size_t i(0); i < res.get_x_dim(); i++)
			for (size_t j(0); j < res.get_y_dim(); j++)
				res[i][j] = h_c[i*res.get_y_dim() + j];
		cudaFree(d_a);
		cudaFree(d_c);
		free(h_a);
		free(h_c);
		return res;
	}

	template<class X>
	friend Matrix<T> operator*(const X& num, const Matrix<T>& a)
	{
		return a*num;
	}

	bool operator!=(const Matrix<T> right)
	{
		return !(*this == right);
	}
};



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
void Matrix<T>::push_back(const Vector<T>& vec)
{
	if (get_y_dim() == vec.size())
		matrix.push_back(vec);
	else
		throw exception("Can't push back due to different sizes of matrix and vector");
}

template <class T>
Matrix<T>::Matrix(const Vector<T>& vec) : Matrix()
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
T Matrix<T>::determinant()
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
