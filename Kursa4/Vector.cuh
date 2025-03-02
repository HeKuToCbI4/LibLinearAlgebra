#pragma once
#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <exception>
#include "Protector.cuh"
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
__global__ void sumVec(const T* a, T* num, size_t size)
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
__global__ void diffKernel(T *c, const T *a, const T *b, size_t N)
{
	size_t i = threadIdx.x * 16 + blockIdx.x*blockDim.x * 16;
	for (size_t j(0); j < 16; j++)
		if (i + j < N)
			c[i + j] = a[i + j] - b[i + j];
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
	Vector& operator +=(const Vector<T>& a);
	Vector& operator -=(const Vector<T>& a);
	Vector operator -(const Vector<T>& a)
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
		h_c = static_cast<T*>(malloc(size));
		cudaMalloc(&d_a, size);
		cudaMalloc(&d_b, size);
		cudaMalloc(&d_c, size);
		cudaMemcpy(d_a, &a[0], size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, &b[0], size, cudaMemcpyHostToDevice);
		size_t threadsperblock = 16;
		size_t blockspergrid = (a.size() + threadsperblock * 16 - 1) / threadsperblock / 16;
		diffKernel<< <blockspergrid, threadsperblock >> > (d_c, d_b, d_a, a.size());
		cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
		result.assign(h_c, h_c + a.size());
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		free(h_c);
		return result;
	}

	friend
		T operator *(const Vector<T>& a, const Vector<T> &b)
	{
		T* d_a;
		T* d_b;
		T* d_c;
		if (a.size() != b.size())
			throw exception("YOBA");
		size_t size = sizeof(T)*a.size();
		cudaMalloc(&d_a, size);
		cudaMalloc(&d_b, size);
		cudaMalloc(&d_c, size);
		cudaMemcpy(d_a, &a[0], size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, &b[0], size, cudaMemcpyHostToDevice);
		size_t threadsperblock = 16;
		size_t blockspergrid = (a.size() + threadsperblock * 16 - 1) / threadsperblock / 16;
		mulKernel << <blockspergrid, threadsperblock >> > (d_a, d_b, d_c, a.size());
		
		T* sum;
		sum = static_cast<T*>(malloc(sizeof(T)));
		T* d_sum;
		cudaMalloc(&d_sum, sizeof(T));
		sumVec << <1, 1 >> > (d_c, d_sum, a.size());
		cudaMemcpy(sum, d_sum, sizeof(T), cudaMemcpyDeviceToHost);
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
		h_c = static_cast<T*>(malloc(size));
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
	friend
	T mixed_multiple(const Vector<T>& vec1, const Vector<T>& vec2, const Vector<T>& vec3)
	{

		Matrix<T> temp;
		if (!(vec1.size() == vec2.size() && vec2.size() == vec3.size() && vec3.size() == 3))
			throw exception("Vector sizes aren't three. NOPE");
		temp.push_back(vec1);
		temp.push_back(vec2);
		temp.push_back(vec3);
		return temp.determinant();
	}

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

	friend istream& operator >> (istream& is, Vector<T>& right)
	{
		right = Vector<T>();
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

	Vector sum_vectors(const Vector<T>& vec)
	{
		///Just for test.///
		Vector<T> res;
		for (auto i(0); i<vec.size(); i++)
		{
			res.emplace_back(vec[i] + this->at(i));
		}
		return res;
	}
	Vector diff_vectors(const Vector<T>& vec)
	{
		///Just for test.///
		Vector<T> res;
		for (auto i(0); i<vec.size(); i++)
		{
			res.emplace_back(vec[i] - this->at(i));
		}
		return res;
	}
	Vector scalar_vectors(const Vector<T>& vec)
	{
		///Just for test.///
		T res=0;
		for (auto i(0); i<vec.size(); i++)
		{
			res+=(vec[i] * this->at(i));
		}
		return res;
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
	h_c = static_cast<T*>(malloc(size));
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	cudaMalloc(&d_c, size);
	cudaMemcpy(d_a, &a[0], size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b[0], size, cudaMemcpyHostToDevice);
	size_t threadsperblock = 16;
	size_t blockspergrid = (a.size() + threadsperblock * 16 - 1) / threadsperblock / 16;
	addKernel << <blockspergrid, threadsperblock >> > (d_c, d_a, d_b, a.size());
	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
	result.assign(h_c, h_c + a.size());
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(h_c);
	return result;
}

template <class T>
Vector<T>& Vector<T>::operator+=(const Vector<T>& a)
{
	Vector<T> b = *this;
	T* d_a;
	T* d_b;

	if (a.size() != b.size())
		throw exception("YOBA");
	size_t size = sizeof(T)*a.size();
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	T* h_c = static_cast<T*>(malloc(size));
	cudaMemcpy(d_a, &a[0], size, cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_b, &b[0], size, cudaMemcpyHostToDevice);
	size_t threadsperblock = 16;
	size_t blockspergrid = (a.size() + threadsperblock * 16 - 1) / threadsperblock / 16;
	addKernel << <blockspergrid, threadsperblock >> > (d_a, d_a, d_b, a.size());
	cudaMemcpy(h_c, d_a, size, cudaMemcpyDeviceToHost);
	this->assign(h_c, h_c + a.size());
	cudaFree(d_a);
	cudaFree(d_b);
	return *this;
}

template <class T>
Vector<T>& Vector<T>::operator-=(const Vector<T>& a)
{
	Vector<T> b = *this;
	T* d_a;
	T* d_b;

	if (a.size() != b.size())
		throw exception("YOBA");
	size_t size = sizeof(T)*a.size();
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	T* h_c = static_cast<T*>(malloc(size));
	cudaMemcpy(d_a, &a[0], size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b[0], size, cudaMemcpyHostToDevice);
	size_t threadsperblock = 16;
	size_t blockspergrid = (a.size() + threadsperblock * 16 - 1) / threadsperblock / 16;
	addKernel << <blockspergrid, threadsperblock >> > (d_a, d_b, d_a, a.size());
	cudaMemcpy(h_c, d_a, size, cudaMemcpyDeviceToHost);
	this->assign(h_c, h_c + a.size());
	cudaFree(d_a);
	cudaFree(d_b);
	return *this;
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

