#pragma once
#include "Vector.cuh"
#include "cuda_runtime.h"
#include <iostream>
#include <exception>

template <class T>
__global__ void matMulKernel(const T* a, const T* b, T* c, size_t ay, size_t by, size_t cy);

template<class T>
__global__ void matCompareKernel(const T* a, const T* b, bool* res, size_t ay);

template <class T>
class Matrix
{
	Vector<Vector<T>> matrix;
	Protector* protector = Protector::get_instance();
public:
	__declspec(dllexport) Matrix();

	__declspec(dllexport) Matrix(size_t x, size_t y);

	__declspec(dllexport) void push_back(Vector<T> vec);

	__declspec(dllexport) Matrix(const Vector<T>& vec);

	__declspec(dllexport) Matrix(const Matrix<T>& mat);

	__declspec(dllexport) double determinant();

	friend
		__declspec(dllexport) double Determinant(Matrix<T> matr)
	{
		if (!matr.get_x_dim() > 0 || !matr.get_y_dim() > 0)
			throw exception("Matrix is not initialized!");
		if (matr.get_x_dim() != matr.get_y_dim())
			throw exception("Matrix is not square!");
		int i, j, j1, j2;
		double det;
		det = 0.0;
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
				det += pow(-1.0, 1.0 + j1 + 1.0) * matr[0][j1] * Determinant(m);
			}
		}
		return(det);

	}

	__declspec(dllexport) Matrix<T> transponate();

	__declspec(dllexport) const Vector<T>& operator[](size_t index) const;

	__declspec(dllexport) Vector<T>& operator[](size_t index);

	__declspec(dllexport) size_t get_x_dim() const;

	__declspec(dllexport) size_t get_y_dim() const;

	friend
		__declspec(dllexport) Matrix<T> operator + (Matrix<T> &a, Matrix<T> &b)
	{
		if (!((a.get_x_dim() == b.get_x_dim()) && (b.get_y_dim() == a.get_y_dim())))
			throw exception("Matrix sizes are different. Can't add them"); \
			Matrix<T> res = Matrix(a.get_x_dim(), a.get_y_dim());
		for (auto i(0); i < a.get_x_dim(); i++)
		{
			res[i] = (a[i] + b[i]);
		}
		return res;
	}
	friend
		__declspec(dllexport) Matrix<T> operator*(const Matrix<T> &a, const Matrix<T> &b)
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

	friend
		__declspec(dllexport) Matrix<T> operator*(const Matrix<T>&a, const Vector<T> &b)
	{
		Matrix<T> tmp(b);

		return a*tmp;
	}

	friend 
		__declspec(dllexport) Matrix<T> operator*(const Vector<T>& b, const Matrix<T>&a)
	{
		Matrix<T> tmp(b);

		return tmp*a;
	}
	friend
		__declspec(dllexport) bool operator==(const Matrix<T> a, const Matrix<T>& b)
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
