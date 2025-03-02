#include "ComplexNumber.cuh"
#include <string>


ComplexNumber::ComplexNumber()
{
	real = 0;
	image = 0;
}

double ComplexNumber::get_real() const
{
	return real;
}

double ComplexNumber::get_imaginary() const
{
	return image;
}

void ComplexNumber::set_real(const double& r)
{
	real = r;
}

void ComplexNumber::set_imaginary(const double& im)
{
	image = im;
}

ComplexNumber& ComplexNumber::operator=(const ComplexNumber& num)
{
	real = num.real;
	image = num.image;
	return *this;
}

ComplexNumber ComplexNumber::operator+(const ComplexNumber& right) const
{
	ComplexNumber res;
	res.real = real + right.real;
	res.image = image + right.image;
	return res;
}

ComplexNumber ComplexNumber::operator*(const ComplexNumber& right) const
{
	ComplexNumber res;
	res.real = real*right.real - image*right.image;
	res.image = real*right.image + image*right.real;
	return res;
}

ComplexNumber ComplexNumber::operator/(const ComplexNumber& num) const
{
	ComplexNumber res(*this);
	res.real = (*this*num.get_conjugation()).get_real() / (num*num.get_conjugation()).get_real();
	res.image = (*this*num.get_conjugation()).get_imaginary() / (num*num.get_conjugation()).get_real();
	return res;
}

ComplexNumber ComplexNumber::operator-(const ComplexNumber& num) const
{
	auto res(*this);
	res.real -= num.real;
	res.image -= num.image;
	return res;
}

bool ComplexNumber::operator==(const ComplexNumber& right) const
{
	return real == right.real && image == right.image;
}

ComplexNumber::ComplexNumber(const ComplexNumber& num)
{
	real = num.real;
	image = num.image;
}

ComplexNumber ComplexNumber::get_conjugation() const
{
	ComplexNumber res(*this);
	res.image = image*(-1);
	return res;
}


double ComplexNumber::module() const
{
	return sqrt(real*real + image*image);
}

double ComplexNumber::argument() const
{
	return atan(image / real);
}

ComplexNumber& ComplexNumber::operator+=(const ComplexNumber& num)
{
	this->real += num.real;
	this->image += num.image;
	return *this;
}

ComplexNumber& ComplexNumber::operator-=(const ComplexNumber& num)
{
	this->real -= num.real;
	this->image -= num.image;
	return *this;
}

ComplexNumber& ComplexNumber::operator*=(const ComplexNumber& num)
{
	ComplexNumber res;
	res.real = real*num.real - image*num.image;
	res.image = real*num.image + image*num.real;
	this->real = res.real;
	this->image = res.image;
	return *this;
}

ComplexNumber& ComplexNumber::operator/=(ComplexNumber& num)
{
	auto res = *this * num.get_conjugation() / (num*num.get_conjugation()).get_real();
	this->real = res.real;
	this->image = res.image;
	return *this;
}

ComplexNumber::~ComplexNumber()
{
}

bool ComplexNumber::operator!=(const ComplexNumber& right) const
{
	return (*this) == right;
}


std::ostream& operator<<(std::ostream& os, const ComplexNumber& num)
{
	os << num.real;
	if (num.image >= 0)
	{
		os << "+" << num.image << "*i";
	}
	else
	{
		os << num.image << "*i";
	}
	return os;
}

std::istream& operator >> (std::istream& is, ComplexNumber& num)
{
	std::string s;
	is >> s;
	if (s.find('+') != std::string::npos)
		sscanf_s(s.c_str(), "%lf+%lf*i", &num.real, &num.image);
	else
		sscanf_s(s.c_str(), "%lf%lf*i", &num.real, &num.image);
	return is;
}