#pragma once
#include <iostream>
#include <cuda_runtime.h>

class ComplexNumber
{
	double real;
	double image;
public:
	__declspec(dllexport)__host__ __device__  ComplexNumber();
	__declspec(dllexport)__host__ __device__ ComplexNumber(const double& real_x, const double& image_x) : real(real_x), image(image_x) {};
	template <class T>
	__declspec(dllexport)__host__ __device__ ComplexNumber(const T& num)
	{
		real = num;
		image = 0;
	}
	__declspec(dllexport)__host__ __device__ double get_real() const;
	__declspec(dllexport)__host__ __device__  double get_imaginary() const;
	__declspec(dllexport)__host__ __device__ void set_real(const double&);
	__declspec(dllexport)__host__ __device__ void set_imaginary(const double&);
	__declspec(dllexport)__host__ __device__ ComplexNumber& operator =(const ComplexNumber&);
	__declspec(dllexport)__host__ __device__ ComplexNumber operator+(const ComplexNumber&) const;
	__declspec(dllexport)__host__ __device__ ComplexNumber operator*(const ComplexNumber&) const;
	__declspec(dllexport)__host__ __device__ ComplexNumber operator/(const ComplexNumber&) const;
	__declspec(dllexport)__host__ __device__ ComplexNumber operator-(const ComplexNumber&) const;
	__declspec(dllexport)__host__ __device__ bool operator ==(const ComplexNumber&) const;
	__declspec(dllexport)__host__ __device__ ComplexNumber(const ComplexNumber&);
	__declspec(dllexport)__host__ __device__  ComplexNumber get_conjugation() const;
	friend
		__declspec(dllexport)std::ostream& operator<<(std::ostream&, const ComplexNumber&);
	friend
		__declspec(dllexport)std::istream& operator >> (std::istream&, ComplexNumber&);

	__declspec(dllexport)__host__ __device__  double module() const;
	__declspec(dllexport)__host__ __device__ double argument() const;
	__declspec(dllexport)__host__ __device__ ComplexNumber& operator+=(const ComplexNumber&);
	__declspec(dllexport)__host__ __device__ ComplexNumber& operator-=(const ComplexNumber&);
	__declspec(dllexport)__host__ __device__ ComplexNumber& operator*=(const ComplexNumber&);
	__declspec(dllexport)__host__ __device__ ComplexNumber& operator/=(ComplexNumber&);
	__declspec(dllexport)__host__ __device__  ~ComplexNumber();
	__declspec(dllexport)__host__ __device__ bool operator!=(const ComplexNumber&) const;
};
