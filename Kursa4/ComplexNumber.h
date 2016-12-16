#pragma once
#include <iostream>
#include "Protector.h"

class ComplexNumber
{
	double real;
	double image;
	Protector* protector = Protector::get_instance();
public:
	__declspec(dllexport) ComplexNumber();
	__declspec(dllexport) ComplexNumber(double real_x, double image_x): real(real_x), image(image_x){};
	template <class T>
	__declspec(dllexport) ComplexNumber(const T& num);
	__declspec(dllexport) double get_real() const;
	__declspec(dllexport) double get_imaginary() const;
	__declspec(dllexport) void set_real(const double&);
	__declspec(dllexport) void set_imaginary(const double&);
	__declspec(dllexport) ComplexNumber& operator =(const ComplexNumber&);
	__declspec(dllexport) ComplexNumber operator+(const ComplexNumber&) const;
	__declspec(dllexport) ComplexNumber operator*(const ComplexNumber&) const;
	__declspec(dllexport) ComplexNumber operator/(const ComplexNumber&) const;
	__declspec(dllexport) ComplexNumber operator-(const ComplexNumber&) const;
	__declspec(dllexport) bool operator ==(const ComplexNumber&) const;
	__declspec(dllexport) ComplexNumber(const ComplexNumber&);
	__declspec(dllexport) ComplexNumber get_conjugation() const;
	friend
		__declspec(dllexport) std::ostream& operator<<(std::ostream&, const ComplexNumber&);
	friend
		__declspec(dllexport) std::istream& operator>>(std::istream&, ComplexNumber&);
	
	__declspec(dllexport) double module() const;
	__declspec(dllexport) double argument() const;
	__declspec(dllexport) ComplexNumber& operator+=(const ComplexNumber&);
	__declspec(dllexport) ComplexNumber& operator-=(const ComplexNumber&);
	__declspec(dllexport) ComplexNumber& operator*=(const ComplexNumber&);
	__declspec(dllexport) ComplexNumber& operator/=(ComplexNumber&);
	__declspec(dllexport) ~ComplexNumber();
	__declspec(dllexport) bool operator!=(const ComplexNumber&) const;
};

template <class T>
ComplexNumber::ComplexNumber(const T& num)
{
	real = num;
	image = 0;
}