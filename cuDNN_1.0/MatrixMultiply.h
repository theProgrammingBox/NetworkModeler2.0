#pragma once
#include "Header.h"

class MatrixMultiply
{
public:
	cublasHandle_t* cublasHandle;
	curandGenerator_t* curandHandle;
	
	size_t mat1Rows, mat1Cols, mat3Cols, batches;
	size_t mat1Size, mat2Size, mat3Size;
	size_t fullMat1Size, fullMat2Size, fullMat3Size;
	size_t fullMat1Bytes, fullMat2Bytes, fullMat3Bytes;
	
	float* gpuMat1, * gpuMat2, * gpuMat3;
	float* cpuMat1, * cpuMat2, * cpuMat3;
	
	MatrixMultiply(cublasHandle_t* cublasHandle, curandGenerator_t* curandHandle, size_t mat1Rows, size_t mat1Cols, size_t mat3Cols, size_t batches) :
		cublasHandle(cublasHandle), curandHandle(curandHandle), mat1Rows(mat1Rows), mat1Cols(mat1Cols), mat3Cols(mat3Cols), batches(batches)
	{
		mat1Size = mat1Cols * mat1Rows;
		mat2Size = mat3Cols * mat1Cols;
		mat3Size = mat3Cols * mat1Rows;
		fullMat1Size = mat1Size * batches;
		fullMat2Size = mat2Size * batches;
		fullMat3Size = mat3Size * batches;
		fullMat1Bytes = fullMat1Size * sizeof(float);
		fullMat2Bytes = fullMat2Size * sizeof(float);
		fullMat3Bytes = fullMat3Size * sizeof(float);
		
		cudaMalloc(&gpuMat1, fullMat1Bytes);
		cudaMalloc(&gpuMat2, fullMat2Bytes);
		cudaMalloc(&gpuMat3, fullMat3Bytes);
		
		cpuMat1 = new float[fullMat1Size];
		cpuMat2 = new float[fullMat2Size];
		cpuMat3 = new float[fullMat3Size];
		
		RandomizeMat1();
		RandomizeMat2();
	}
	
	~MatrixMultiply()
	{
		cudaFree(gpuMat1);
		cudaFree(gpuMat2);
		cudaFree(gpuMat3);

		delete[] cpuMat1;
		delete[] cpuMat2;
		delete[] cpuMat3;
	}

	void Mat1MulMat2()
	{
		cublasSgemmStridedBatched(
			*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			mat3Cols, mat1Rows, mat1Cols,
			&ONEf, gpuMat2, mat3Cols, mat2Size,
			gpuMat1, mat1Cols, mat1Size,
			&ZEROf, gpuMat3, mat3Cols, mat3Size,
			batches);
		
		cudaMemcpy(cpuMat1, gpuMat1, fullMat1Bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpuMat2, gpuMat2, fullMat2Bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpuMat3, gpuMat3, fullMat3Bytes, cudaMemcpyDeviceToHost);
		
		float error = 0;
		for (size_t batch = 0; batch < batches; batch++)
		{
			for (size_t row = 0; row < mat1Rows; row++)
			{
				for (size_t col = 0; col < mat3Cols; col++)
				{
					float sum = 0;
					for (size_t i = 0; i < mat1Cols; i++)
					{
						sum += cpuMat1[batch * mat1Size + row * mat1Cols + i] * cpuMat2[batch * mat2Size + i * mat3Cols + col];
					}
					error += abs(sum - cpuMat3[batch * mat3Size + row * mat3Cols + col]);
				}
			}
		}
		cout << "Mat1MulMat2 error: " << error / (mat1Rows * mat3Cols * batches) << "\n";
	}

	void Mat1TMulMat3()
	{
		cublasSgemmStridedBatched(
			*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
			mat3Cols, mat1Cols, mat1Rows,
			&ONEf, gpuMat3, mat3Cols, mat3Size,
			gpuMat1, mat1Cols, mat1Size,
			&ZEROf, gpuMat2, mat3Cols, mat2Size,
			batches);

		cudaMemcpy(cpuMat1, gpuMat1, fullMat1Bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpuMat2, gpuMat2, fullMat2Bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpuMat3, gpuMat3, fullMat3Bytes, cudaMemcpyDeviceToHost);

		float error = 0;
		for (size_t batch = 0; batch < batches; batch++)
		{
			for (size_t row = 0; row < mat1Cols; row++)
			{
				for (size_t col = 0; col < mat3Cols; col++)
				{
					float sum = 0;
					for (size_t i = 0; i < mat1Rows; i++)
					{
						sum += cpuMat1[batch * mat1Size + i * mat1Cols + row] * cpuMat3[batch * mat3Size + i * mat3Cols + col];
					}
					error += abs(sum - cpuMat2[batch * mat2Size + row * mat3Cols + col]);
				}
			}
		}
		cout << "Mat1TMulMat3 error: " << error / (mat1Cols * mat3Cols * batches) << "\n";
	}

	void Mat3MulMat2T()
	{
		cublasSgemmStridedBatched(
			*cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
			mat1Cols, mat1Rows, mat3Cols,
			&ONEf, gpuMat2, mat3Cols, mat2Size,
			gpuMat3, mat3Cols, mat3Size,
			&ZEROf, gpuMat1, mat1Cols, mat1Size,
			batches);

		cudaMemcpy(cpuMat1, gpuMat1, fullMat1Bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpuMat2, gpuMat2, fullMat2Bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpuMat3, gpuMat3, fullMat3Bytes, cudaMemcpyDeviceToHost);

		float error = 0;
		for (size_t batch = 0; batch < batches; batch++)
		{
			for (size_t row = 0; row < mat1Rows; row++)
			{
				for (size_t col = 0; col < mat1Cols; col++)
				{
					float sum = 0;
					for (size_t i = 0; i < mat3Cols; i++)
					{
						sum += cpuMat2[batch * mat2Size + col * mat3Cols + i] * cpuMat3[batch * mat3Size + row * mat3Cols + i];
					}
					error += abs(sum - cpuMat1[batch * mat1Size + row * mat1Cols + col]);
				}
			}
		}
		cout << "Mat3MulMat2T error: " << error / (mat1Rows * mat1Cols * batches) << "\n";
	}

	void RandomizeMat1()
	{
		curandGenerateUniform(*curandHandle, gpuMat1, fullMat1Size);
	}

	void RandomizeMat2()
	{
		curandGenerateUniform(*curandHandle, gpuMat2, fullMat2Size);
	}

	void RandomizeMat3()
	{
		curandGenerateUniform(*curandHandle, gpuMat3, fullMat3Size);
	}

	void PrintMat1()
	{
		cudaMemcpy(cpuMat1, gpuMat1, fullMat1Bytes, cudaMemcpyDeviceToHost);

		cout << "Mat1:\n";
		for (size_t batch = 0; batch < batches; batch++)
		{
			for (size_t row = 0; row < mat1Rows; row++)
			{
				for (size_t col = 0; col < mat1Cols; col++)
				{
					cout << cpuMat1[batch * mat1Size + row * mat1Cols + col] << " ";
				}
				cout << "\n";
			}
			cout << "\n";
		}
		cout << "\n";
	}
	
	void PrintMat2()
	{
		cudaMemcpy(cpuMat2, gpuMat2, fullMat2Bytes, cudaMemcpyDeviceToHost);

		cout << "Mat2:\n";
		for (size_t batch = 0; batch < batches; batch++)
		{
			for (size_t row = 0; row < mat1Cols; row++)
			{
				for (size_t col = 0; col < mat3Cols; col++)
				{
					cout << cpuMat2[batch * mat2Size + row * mat3Cols + col] << " ";
				}
				cout << "\n";
			}
			cout << "\n";
		}
		cout << "\n";
	}

	void PrintMat3()
	{
		cudaMemcpy(cpuMat3, gpuMat3, fullMat3Bytes, cudaMemcpyDeviceToHost);

		cout << "Mat3:\n";
		for (size_t batch = 0; batch < batches; batch++)
		{
			for (size_t row = 0; row < mat1Rows; row++)
			{
				for (size_t col = 0; col < mat3Cols; col++)
				{
					cout << cpuMat3[batch * mat3Size + row * mat3Cols + col] << " ";
				}
				cout << "\n";
			}
			cout << "\n";
		}
		cout << "\n";
	}

	void SaveToFile(string fileName)
	{
		cudaMemcpy(cpuMat1, gpuMat1, fullMat1Bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpuMat2, gpuMat2, fullMat2Bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpuMat3, gpuMat3, fullMat3Bytes, cudaMemcpyDeviceToHost);

		ofstream file(fileName, ios::out | ios::binary);
		file.write((char*)cpuMat1, fullMat1Bytes);
		file.write((char*)cpuMat2, fullMat2Bytes);
		file.write((char*)cpuMat3, fullMat3Bytes);
		file.close();
	}

	void LoadFromFile(string fileName)
	{
		ifstream file(fileName, ios::in | ios::binary);
		file.read((char*)cpuMat1, fullMat1Bytes);
		file.read((char*)cpuMat2, fullMat2Bytes);
		file.read((char*)cpuMat3, fullMat3Bytes);
		file.close();

		cudaMemcpy(gpuMat1, cpuMat1, fullMat1Bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(gpuMat2, cpuMat2, fullMat2Bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(gpuMat3, cpuMat3, fullMat3Bytes, cudaMemcpyHostToDevice);
	}
};