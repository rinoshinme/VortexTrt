#pragma once
#include <string>
#include <memory.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "NvInfer.h"

namespace vortex
{

    enum class MemoryType
    {
        GPU_MEMORY = 0,
        CPU_MEMORY,
    };

    struct BlobInfo
    {
        std::string name;
        uint32_t batch;
        uint32_t channels;
        uint32_t height;
        uint32_t width;
        uint32_t Count() const { return batch * channels * height * width; }
    };

    template<typename T>
    struct Blob
    {
        std::string name;
        T* dataCpu;
        T* dataGpu;
        uint32_t count;

        Blob(const BlobInfo& info)
        {
            this->name = info.name;
            this->count = info.Count();
            dataCpu = new T[count];
            checkRuntime(cudaMalloc(&dataGpu, count * sizeof(T)));
        }

        Blob(const std::string& name, uint32_t count)
        {
            this->name = name;
            this->count = count;
            dataCpu = new T[count];
            checkRuntime(cudaMalloc(&dataGpu, count * sizeof(T)));
        }

        ~Blob()
        {
            delete dataCpu;
            dataCpu = nullptr;
            checkRuntime(cudaFree(dataGpu));
            dataGpu = nullptr;
        }

        void ToGpu()
        {
            checkRuntime(cudaMemcpy(dataGpu, dataCpu, count * sizeof(T), cudaMemcpyHostToDevice));
        }

        void ToCpu()
        {
            checkRuntime(cudaMemcpy(dataCpu, dataGpu, count * sizeof(T), cudaMemcpyDeviceToHost));
        }

        void ToGpuAsync(cudaStream_t stream, T* source = nullptr)
        {
            if (source == nullptr)
                source = dataCpu;
            checkRuntime(cudaMemcpyAsync(dataGpu, source, count * sizeof(T), cudaMemcpyHostToDevice, stream));
        }

        void ToCpuAsync(cudaStream_t stream, T* target = nullptr)
        {
            if (target == nullptr)
                target = dataCpu;
            checkRuntime(cudaMemcpyAsync(target, dataGpu, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
        }

        void CopyFrom(const T* data, uint32_t size = 0)
        {
            if (size == 0)
                size = count;
            memcpy(dataCpu, data, size * sizeof(T));
        }

        void CopyTo(T* data, uint32_t size = 0)
        {
            if (size == 0)
                size = count;
            memcpy(data, dataCpu, size * sizeof(T));
        }
    };

    typedef Blob<float> BlobF;
    typedef Blob<uint8_t> BlobI;
}
