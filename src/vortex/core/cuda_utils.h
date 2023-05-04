/*
cuda and tensorrt utility functions
*/
#pragma once
#include <cstdio>
#include <cuda_runtime.h>



namespace vortex
{
#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

    bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line)
    {
        if (code != cudaSuccess)
        {
            const char* err_name = cudaGetErrorName(code);
            const char* err_message = cudaGetErrorString(code);
            printf("runtime error: %s:%d %s failed.\n code = %s, message = %s\n", file, line, op, err_name, err_message);
            return false;
        }
        return true;
    }

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
}
