#include "MatMulMat.h"

int main()
{
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	curandGenerator_t curandHandle;
	curandCreateGenerator(&curandHandle, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(curandHandle, 1234ULL);

	const size_t BATCH_SIZE = 2;
	const size_t INPUT_SIZE = 4;
	const size_t INPUT_FEATURES = 3;
	const size_t OUTPUT_FEATURES = 2;

	MatMulMat matMulMat(&cublasHandle, &curandHandle, INPUT_SIZE, INPUT_FEATURES, OUTPUT_FEATURES, BATCH_SIZE);
	matMulMat.Forward();

	matMulMat.PrintMat1();
	matMulMat.PrintMat2();
	matMulMat.PrintMat3();

	matMulMat.RandomizeMat1();
	matMulMat.Mat2Backward();
	matMulMat.PrintMat1();
	matMulMat.PrintMat3();
	matMulMat.PrintMat2();

	matMulMat.Mat1Backward();
	matMulMat.PrintMat3();
	matMulMat.PrintMat2();
	matMulMat.PrintMat1();
	
	curandDestroyGenerator(curandHandle);
	cublasDestroy(cublasHandle);
		
	return 0;
}