#include "ComplexNumber.h"



ComplexNumber::ComplexNumber()
{
	real = 0;
	image = 0;
}

double ComplexNumber::get_real()
{
	return real;
}

double ComplexNumber::get_imaginary()
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

ComplexNumber ComplexNumber::operator+(const ComplexNumber& right)
{
	ComplexNumber res;
	res.real = real + right.real;
	res.image = image + right.image;
	return res;
}

ComplexNumber ComplexNumber::operator*(const ComplexNumber& right)
{
	ComplexNumber res;
	res.real = real*right.real - image*right.image;
	res.image = real*res.image + image*right.real;
	return res;
}

ComplexNumber ComplexNumber::operator/(const ComplexNumber&)
{
	return get_conjugation() / (*this*get_conjugation()).get_real();
}

ComplexNumber ComplexNumber::operator-(const ComplexNumber& num)
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

ComplexNumber::~ComplexNumber()
{
}

template <class T>
ComplexNumber ComplexNumber::operator/(const T& num)
{
	auto res(*this);
	res.image /= num;
	res.real /= num;
	return res;
}

template <class T>
ComplexNumber ComplexNumber::operator+(const T& num)
{
	auto res(*this);
	res.real += num;
	return res;
}



template <class T>
ComplexNumber ComplexNumber::operator-(const T& num)
{
	auto res(*this);
	res.real -= num;
	return res;
}

template <class T>
ComplexNumber ComplexNumber::operator*(const T& num)
{
	auto res(*this);
	res.real *= num;
	res.image *= num;
	return res;
}

template <class T>
ComplexNumber& ComplexNumber::operator+=(const T& num)
{
	*this = *this + num;
	return *this;
}

std::ostream& operator<<(std::ostream& os, const ComplexNumber& num)
{
	os << num.real;
	if (num.image>0)
	{
		os << "+" << num.image << "i";
	} 
	else if (num.image!=0)
	{
		os << num.image << "i";
	}
	return os;
 }

std::istream& operator>>(std::istream& is, ComplexNumber& num)
{
	std::string s;
	is >> s;
	if (s.find('+') != std::string::npos)
		sscanf_s(s.c_str(), "%lf+%lfi", &num.real, &num.image);
	else
		sscanf_s(s.c_str(), "%lf%lfi", &num.real, &num.image);
	return is;
}