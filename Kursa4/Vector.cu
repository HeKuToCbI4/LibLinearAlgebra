
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <exception>
#include "ComplexNumber.h"
using namespace std;

template <class T>
__global__ void addKernel(T *c, const T *a, const T *b, size_t N)
{
	size_t i = threadIdx.x*16 + blockIdx.x*blockDim.x*16;
	for (size_t j(0); j < 16; j++)
		if (i + j < N)
			c[i + j] = a[i + j] + b[i + j];
}
template <class T>
__global__ void mulKernel(const T* a, const T* b, T *c, size_t N)
{
	size_t i = threadIdx.x*16 + blockIdx.x*blockDim.x*16;
	for (size_t j(0); j < 16; j++)
		if (i + j < N)
		c[i+j] = a[i+j] * b[i+j];
}

template <class T, class X>
__global__ void mulByNum(const T*a, T* b, const X n, size_t N)
{
	size_t i = threadIdx.x*16 + blockIdx.x*blockDim.x*16;
	for (char j = 0; j < 16; j++)
		if (i + j < N)
			b[i + j] = a[i + j] * n;
}
template <class T>
__global__ void sumVec(const T* a, double* num, size_t size)
{
	num[0] = 0;
	for (size_t i(0); i < size; i++)
		num[0] += a[i];
}




template <class T>
class Vector : public vector<T>
{
public:
	Vector<T>() : vector<T>()
	{
	}
	Vector<T>(size_t size) : vector<T>(size)
	{
	}
	Vector operator +(const Vector<T>& a)
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
		h_c = (T*)malloc(size);
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
		addKernel << <blockspergrid, threadsperblock >> > (d_c, d_a, d_b, a.size());
		if (cudaSuccess != cudaGetLastError())
		{
			cout << "Error in kernel!\n";
			getchar();
		}
		if (cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost) != cudaSuccess)
		{
			cout << "error in copying memory from device to host\n";
			getchar();
		}
		result.assign(h_c, h_c + a.size());
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		free(h_c);
		return result;
	}
	friend double operator *(const Vector<T>& a, const Vector<T> &b)
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
		mulKernel <<<blockspergrid, threadsperblock >>> (d_a, d_b, d_c, a.size());
		if (cudaSuccess != cudaGetLastError())
		{
			cout << "Error in kernel!\n";
			getchar();
		}
		double* sum;
		sum = (double*)malloc(sizeof(double));
		double* d_sum;
		cudaMalloc(&d_sum, sizeof(double));
		sumVec<<<1,1>>>(d_c, d_sum, a.size());
		cudaMemcpy(sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		return sum[0];
	}
	template <class X>
	friend Vector operator *(const Vector<T>& a, const X& b)
	{
		Vector<T> result = Vector(a.size());
		T* d_a;
		T* d_b;
		T* h_c;
		size_t size = sizeof(T)*a.size();
		h_c = (T*)malloc( size);
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
	friend Vector operator *(const X& b, const Vector<T>& a)
	{
		return a*b;
	}
	Vector cross_multiple(const Vector<T>&);
	Vector mixed_multiple(const Vector<T>&);
};



template <class T>
Vector<T> Vector<T>::cross_multiple(const Vector<T>&)
{
}

template <class T>
Vector<T> Vector<T>::mixed_multiple(const Vector<T>&)
{
}

template <class T>
void print_vec(Vector<T> vec)
{
	for (auto e : vec)
		cout << e << " ";
	cout << endl;
}

template <class T>
class Matrix;

template <class T>
__global__ void mat_sum_kernel(Matrix<T> matrix1, Matrix<T> & matrix2, Matrix<T> matrix_res, size_t x_dim, size_t y_dim)
{
	size_t x = threadIdx.x + blockIdx.x*blockDim.x;
	size_t y = threadIdx.y + blockIdx.y*blockDim.y;
	size_t i = 0;
	size_t j = 0;
		if (i + x < x_dim && j + y < y_dim)
			matrix_res[x][y] = matrix1[x][y]+matrix2[x][y];
}

template <class T>
class Matrix
{
	Vector<Vector<T>> matrix;
public:
	Matrix()
	{
		matrix = Vector<Vector<T>>();
	}
	Matrix(size_t x, size_t y)
	{
		matrix = Vector<Vector<T>>(x);
		for (size_t i(0); i < x; i++)
			matrix[i] = Vector<T>(y);
	}
	void push_back(Vector<T> vec)
	{
		matrix.push_back(vec);
	}
	//typedef Matrix::iterator iterator;
	//typedef Matrix::const_iterator const_itreator;
	//Matrix::iterator begin() { return matrix.begin(); }
	//Matrix::iterator end() { return matrix.end(); }

	double determinant()
	{
		if (!matrix.size() > 0 || !matrix[0].size > 0)
			throw exception("Matrix is not initialized!");
		if (matrix.size() != matrix[0].size())
			throw exception("Matrix is not square!");
	}

	Vector<T>& operator[](size_t index) const
	{
		return matrix[index];
	}
	Vector<T>& operator[](size_t index)
	{
		return matrix[index];
	}
	size_t get_x_dim() const
	{
		return matrix.size();
	}
	size_t get_y_dim() const
	{
		return matrix[0].size();
	}
	friend Matrix<T> operator + (Matrix<T> &a, Matrix<T> &b)
	{
		if (!((a.get_x_dim() == b.get_x_dim()) && (b.get_y_dim() == a.get_y_dim())))
			throw exception("Matrix sizes are different. Can't add them");\
		Matrix<T> res = Matrix(a.get_x_dim(), a.get_y_dim());
		for (auto i(0); i < a.get_x_dim(); i++)
		{
			res[i] = (a[i] + b[i]);
		}
		return res;
	}

};



template <class T>
void print_matr(Matrix<T> matr)
{
	for (auto i(0); i < matr.get_x_dim(); i++)
		print_vec(matr[i]);
	getchar();
}
int main()
{
	ComplexNumber num1(1, 1), num2(2, -3);
	std::cout << num1 << "   " << num2 << endl;
	cin >> num1 >> num2;
	
	cout << "sum " << num1 + num2 << endl;
	cout << "diff " << num1 - num2 << endl;
	cout << "multiple " << num1*num2 << endl;
	cout << "division " << num1 / num2 << endl;
	cout << "conjugation " << num1.get_conjugation() << endl;
	cout << "module " << num1.module() << endl;
	cout << "argument " << num1.argument() << endl;
	getchar();
	Vector<int> vec1, vec2, vec3;
	for (int i = 0; i < 15; i++)
	{
		vec1.emplace_back(1*i);
		vec2.emplace_back(1.124*i*i);
	}
	Matrix<int> matr;
	for (auto i = 0; i < 15; i++)
		matr.push_back(vec1);
	print_matr(matr);
	Matrix<int> matr2 = matr + matr;
	print_matr(matr2);
	//int* mem = (int*)malloc(vec1.size()*sizeof(int));
	//mem = &vec1[0];
	vec3 = vec1 + vec1;
	double scalar_multiple = vec1*vec2;
	print_vec(vec3);
	cout << "SCALAR: " << scalar_multiple << endl;
	getchar();
	return 0;
}

/*cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> >(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}*/