#include "MatMulMat.h"

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

	matMulMat.SaveToFile("MatMulMat.txt");

	MatMulMat matMulMat2(&cublasHandle, &curandHandle, INPUT_SIZE, INPUT_FEATURES, OUTPUT_FEATURES, BATCH_SIZE);
	matMulMat2.LoadFromFile("MatMulMat.txt");
	matMulMat2.PrintMat3();
	matMulMat2.PrintMat2();
	matMulMat2.PrintMat1();
	
	curandDestroyGenerator(curandHandle);
	cublasDestroy(cublasHandle);
		
	return 0;
}