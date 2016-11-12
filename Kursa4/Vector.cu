#include "Vector.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template <class T>
__global__ void addKernel(T *c, const T *a, const T *b, const unsigned int &N)
{
	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i<N)
		c[i] = a[i] + b[i];
}
template <class T>
__global__ void mulKernel(const T* a, const T *b, T *c, const unsigned int &N)
{
	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i<N)
		c[i] = a[i] * b[i];
}
template <class T, class X>
__global__ void mulByNum(const T*a, const X n, T* b, const unsigned int &N)
{
	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < N)
		b[i] = a[i] * n;
}

template <class T>
Vector<T>::Vector()
{
}

template <class T>
Vector<T>::~Vector()
{
}

template <class T>
Vector<T> Vector<T>::operator+(const Vector<T>& a, const Vector<T>& b)
{
	Vector<T> result;
	T* d_a;
	T* d_b;
	T* d_c;
	T* h_c;
	if (a.length() != b.length)
		return nullptr;
	unsigned int size = sizeof(T)*a.length();
	malloc(&h_c, size);
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	cudaMalloc(&d_c, size);
	cudaMemcpy(d_a, &a[0], size, cudaMemcpyHostToDevice); 
	cudaMemcpy(d_b, &b[0], size, cudaMemcpyHostToDevice);
	unsigned int threadsperblock = 256;
	unsigned int blockspergrid = (a.length() + threadsperblock - 1) / threadsperblock;
	addKernel << <blockspergrid, threadsperblock >> > (d_a, d_b, d_c, size);
	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
	result.assign(h_c, h_c + a.length());
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(h_c);
	return result;
}

template <class T>
float Vector<T>::operator*(const Vector<T>&, const Vector<T>&)
{
}

template <class T>
Vector<T> Vector<T>::cross_multiple(const Vector<T>&)
{
}

template <class T>
Vector<T> Vector<T>::mixed_multiple(const Vector<T>&)
{
}

template <class T, class X>
Vector<T> operator *(const Vector<T>& a, const X& b)
{
	Vector<T> result;
	T* d_a;
	T* d_b;
	T* h_c;
	T d_x;
	cudaMalloc(&d_x, sizeof(T));
	cudaMemCpy(d_x, b, sizeof(T), cudaMemcpyHostToDevice);
	unsigned int size = sizeof(T)*a.length();
	malloc(&h_c, size);
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	cudaMemcpy(d_a, &a[0], size, cudaMemcpyHostToDevice);
	unsigned int threadsperblock = 256;
	unsigned int blockspergrid = (a.length() + threadsperblock - 1) / threadsperblock;
	mulByNum<<<blockspergrid, threadsperblock >>> (d_a, d_x, size);
	cudaMemcpy(h_c, d_b, size, cudaMemcpyDeviceToHost);
	result.assign(h_c, h_c + a.length());
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_x);
	free(h_c);
	return result;
}

/*cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
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