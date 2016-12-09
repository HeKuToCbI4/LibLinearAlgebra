#pragma once
#include <iostream>
#include <math.h>
class ComplexNumber
{
	double real;
	double image;
public:
	ComplexNumber();
	ComplexNumber(double real_x, double image_x): real(real_x), image(image_x){};
	double get_real();
	double get_imaginary();
	void set_real(const double&);
	void set_imaginary(const double&);
	ComplexNumber& operator =(const ComplexNumber&);
	ComplexNumber operator+(const ComplexNumber&);
	ComplexNumber operator*(const ComplexNumber&);
	ComplexNumber operator/(const ComplexNumber&);
	bool operator ==(const ComplexNumber&);
	ComplexNumber(const ComplexNumber&);
	ComplexNumber get_conjugation();
	template <class T>
	ComplexNumber operator/(const T&);
	template <class T>
	ComplexNumber operator+(const T&);
	template <class T>
	ComplexNumber operator-(const T&);
	template <class T>
	ComplexNumber operator*(const T&);
	friend
	std::ostream& operator<<(std::ostream&, const ComplexNumber&);
	friend
	std::istream& operator>>(std::istream&, ComplexNumber&);
	template <class T>
	ComplexNumber& operator+=(const T&);
	double module();
	double argument();
	~ComplexNumber();
};



