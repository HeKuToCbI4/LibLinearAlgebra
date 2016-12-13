#pragma once
#include <iostream>
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
	ComplexNumber operator-(const ComplexNumber&);
	bool operator ==(const ComplexNumber&) const;
	ComplexNumber(const ComplexNumber&);
	ComplexNumber get_conjugation() const;
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
	
	double module() const;
	double argument() const;
	ComplexNumber& operator+=(const ComplexNumber&);
	ComplexNumber& operator-=(const ComplexNumber&);
	ComplexNumber& operator*=(const ComplexNumber&);
	ComplexNumber& operator/=(const ComplexNumber&);
	template <class T>
	ComplexNumber& operator+=(const T&);
	template <class T>
	ComplexNumber& operator-=(const T&);
	template <class T>
	ComplexNumber& operator*=(const T&);
	template <class T>
	ComplexNumber& operator/=(const T&);
	~ComplexNumber();
};



