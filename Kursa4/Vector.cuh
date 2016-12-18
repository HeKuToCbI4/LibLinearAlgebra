#pragma once
#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <exception>
#include "Protector.h"
#include "ComplexNumber.h"
#include "device_launch_parameters.h"
#include "Matrix.cuh"

using namespace std;

template <class T>
__global__ void mulKernel(const T* a, const T* b, T* c, size_t N)
{
	size_t i = threadIdx.x * 16 + blockIdx.x*blockDim.x * 16;
	for (size_t j(0); j < 16; j++)
		if (i + j < N)
			c[i + j] = a[i + j] * b[i + j];
}

template <class T, class X>
__global__ void mulByNum(const T* a, T* b, const X n, size_t N)
{
	size_t i = threadIdx.x * 16 + blockIdx.x*blockDim.x * 16;
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
__global__ void addKernel(T *c, const T *a, const T *b, size_t N)
{
	size_t i = threadIdx.x * 16 + blockIdx.x*blockDim.x * 16;
	for (size_t j(0); j < 16; j++)
		if (i + j < N)
			c[i + j] = a[i + j] + b[i + j];
}


template <class T>
class Vector : public vector<T>
{
	Protector* protector = Protector::get_instance();
public:
	 Vector<T>();

	 Vector<T>(size_t size);

	 Vector<T>(const Vector<T>& vec);

	 Vector<T>& operator=(const Vector<T>& vec);

	 Vector operator +(const Vector<T>& a);

	Vector operator -(const Vector<T>& right)
	{
		return -1*right + *this;
	}

	friend 
		 double operator *(const Vector<T>& a, const Vector<T> &b)
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
		mulKernel << <blockspergrid, threadsperblock >> > (d_a, d_b, d_c, a.size());
		if (cudaSuccess != cudaGetLastError())
		{
			cout << "Error in kernel!\n";
			getchar();
		}
		double* sum;
		sum = (double*)malloc(sizeof(double));
		double* d_sum;
		cudaMalloc(&d_sum, sizeof(double));
		sumVec << <1, 1 >> >(d_c, d_sum, a.size());
		cudaMemcpy(sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		return sum[0];
	}
	template <class X>
	friend 
		 Vector operator *(const Vector<T>& a, const X& b)
	{
		Vector<T> result = Vector(a.size());
		T* d_a;
		T* d_b;
		T* h_c;
		size_t size = sizeof(T)*a.size();
		h_c = (T*)malloc(size);
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
	friend 
		 Vector operator *(const X& b, const Vector<T>& a)
	{
		return a*b;
	}
	 double mixed_multiple(const Vector<T>&);

	 bool operator==(const Vector<T>&);
	~Vector() {};

	 friend ostream& operator<<(ostream& os, const Vector<T> right)
	{
		os << right.size() << " ";
		for (const auto& elem : right)
			os << elem << " ";
		cout << endl;
		return os;

	}

	 friend istream& operator>>(istream& is, Vector<T>& right)
	{
		size_t size;
		is >> size;
		for (auto i(0); i < size; i++)
		{
			T tmp;
			is >> tmp;
			right.emplace_back(tmp);
		}
		return is;
	}
};

template <class T>
Vector<T>::Vector() = default;

template <class T>
Vector<T>::Vector(size_t size) : vector<T>(size)
{
}

template <class T>
Vector<T>::Vector(const Vector<T>& vec) = default;

template <class T>
Vector<T>& Vector<T>::operator=(const Vector<T>& vec) = default;

template <class T>
Vector<T> Vector<T>::operator+(const Vector<T>& a)
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

template <class T>
double Vector<T>::mixed_multiple(const Vector<T>& vec)
{
	Matrix<T> temp;
	for (size_t i(0); i < vec.size(); i++)
		temp.push_back(vec);
	return temp.determinant();
}

template <class T>
bool Vector<T>::operator==(const Vector<T>& right)
{
	if ((*this).size() != right.size())
		return false;
	for (auto i = 0; i < right.size(); i++)
		if ((*this)[i] != right[i])
			return false;
	return true;
}

