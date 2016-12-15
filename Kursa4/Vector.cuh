#pragma once
#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <exception>
#include "Protector.h"
using namespace std;

template <class T>
__global__ void mulKernel(const T* a, const T* b, T* c, size_t N);

template <class T, class X>
__global__ void mulByNum(const T* a, T* b, const X n, size_t N);
template <class T>
__global__ void sumVec(const T* a, double* num, size_t size);


template <class T>
class Vector : public vector<T>
{
	Protector* protector = Protector::get_instance();
public:
	__declspec(dllexport) Vector<T>();

	__declspec(dllexport) Vector<T>(size_t size);

	__declspec(dllexport) Vector<T>(const Vector<T>& vec);

	__declspec(dllexport) Vector<T>& operator=(const Vector<T>& vec);

	__declspec(dllexport) Vector operator +(const Vector<T>& a);

	friend 
		__declspec(dllexport) double operator *(const Vector<T>& a, const Vector<T> &b)
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
		__declspec(dllexport) Vector operator *(const Vector<T>& a, const X& b)
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
		__declspec(dllexport) Vector operator *(const X& b, const Vector<T>& a)
	{
		return a*b;
	}
	__declspec(dllexport) double mixed_multiple(const Vector<T>&);
};
