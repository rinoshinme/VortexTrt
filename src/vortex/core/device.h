#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <map>

#include <opencv2/core/core.hpp>
#include "cuda_utils.h"
#include "NvInfer.h"


namespace vortex
{
    enum class Device
    {
        CPU,
        GPU,
    };

    // TODO: provide more functionalities.
    class DeviceMemory
    {
    private:
        float* m_Data;
        uint32_t m_Size;
        
    public:
        DeviceMemory(uint32_t size)
            : m_Size(size)
        {
            checkRuntime(cudaMalloc(&m_Data, size * sizeof(float)));
        }

        ~DeviceMemory()
        {
            checkRuntime(cudaFree(m_Data));
        }

        float* Ptr() const { return m_Data; }
        float* Data() const { return m_Data; }
        uint32_t Size() const { return m_Size; }
    };
}
