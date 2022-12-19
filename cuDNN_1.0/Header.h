#pragma once
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <string>

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::ofstream;
using std::ifstream;
using std::ios;
using std::string;

const float ONEf = 1.0f;
const float ZEROf = 0.0f;