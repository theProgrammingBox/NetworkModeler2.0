#include "MatrixMultiply.h"

int main()
{
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	curandGenerator_t curandHandle;
	curandCreateGenerator(&curandHandle, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(curandHandle, duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());

	const size_t BATCH_SIZE = 2;
	const size_t INPUT_SIZE = 4;
	const size_t INPUT_FEATURES = 3;
	const size_t OUTPUT_FEATURES = 2;

	MatrixMultiply model(&cublasHandle, &curandHandle, INPUT_SIZE, INPUT_FEATURES, OUTPUT_FEATURES, BATCH_SIZE);
	model.RandomizeMat1();
	model.RandomizeMat2();
	model.Mat1MulMat2();
	model.PrintMat1();
	model.PrintMat2();
	model.PrintMat3();

	model.RandomizeMat1();
	model.RandomizeMat3();
	model.Mat1TMulMat3();
	model.PrintMat1();
	model.PrintMat3();
	model.PrintMat2();

	model.RandomizeMat3();
	model.RandomizeMat2();
	model.Mat3MulMat2T();
	model.PrintMat3();
	model.PrintMat2();
	model.PrintMat1();

	model.SaveToFile("MatrixMultiply.txt");

	MatrixMultiply matMulMat2(&cublasHandle, &curandHandle, INPUT_SIZE, INPUT_FEATURES, OUTPUT_FEATURES, BATCH_SIZE);
	matMulMat2.LoadFromFile("MatrixMultiply.txt");
	matMulMat2.PrintMat3();
	matMulMat2.PrintMat2();
	matMulMat2.PrintMat1();
	
	curandDestroyGenerator(curandHandle);
	cublasDestroy(cublasHandle);
		
	return 0;
}