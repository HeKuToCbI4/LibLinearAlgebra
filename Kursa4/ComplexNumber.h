#pragma once
#include <iostream>
#include "Protector.h"

class ComplexNumber
{
	double real;
	double image;
	Protector* protector = Protector::get_instance();
public:
	ComplexNumber();
	ComplexNumber(double real_x, double image_x): real(real_x), image(image_x){};
	template <class T>
	ComplexNumber(const T& num);
	double get_real() const;
	double get_imaginary() const;
	void set_real(const double&);
	void set_imaginary(const double&);
	ComplexNumber& operator =(const ComplexNumber&);
	ComplexNumber operator+(const ComplexNumber&) const;
	ComplexNumber operator*(const ComplexNumber&) const;
	ComplexNumber operator/(const ComplexNumber&) const;
	ComplexNumber operator-(const ComplexNumber&) const;
	bool operator ==(const ComplexNumber&) const;
	ComplexNumber(const ComplexNumber&);
	ComplexNumber get_conjugation() const;
	friend
	std::ostream& operator<<(std::ostream&, const ComplexNumber&);
	friend
	std::istream& operator>>(std::istream&, ComplexNumber&);
	
	double module() const;
	double argument() const;
	ComplexNumber& operator+=(const ComplexNumber&);
	ComplexNumber& operator-=(const ComplexNumber&);
	ComplexNumber& operator*=(const ComplexNumber&);
	ComplexNumber& operator/=(ComplexNumber&);
	~ComplexNumber();
};

template <class T>
ComplexNumber::ComplexNumber(const T& num)
{
	real = num;
	image = 0;
}
