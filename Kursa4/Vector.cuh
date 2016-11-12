#pragma once
#include <Vector>

template <class T>
class Vector :
	public std::Vector<T>
{
public:
	Vector();
	~Vector();
	Vector operator +(const Vector<T>&, const Vector<T>&);
	float operator *(const Vector<T>&, const Vector<T>&);
	template <class X>
	friend Vector operator *(const Vector<T>&, const X&);
	Vector cross_multiple(const Vector<T>&);
	Vector mixed_multiple(const Vector<T>&);
};


