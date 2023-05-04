/*
cuda and tensorrt utility functions
*/
#pragma once
#include <cstdio>
#include <cuda_runtime.h>


bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line);

#ifndef checkRuntime
#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)
#endif

#ifndef CHECK
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)
#endif
