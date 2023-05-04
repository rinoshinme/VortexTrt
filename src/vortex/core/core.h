#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <map>

#include <opencv2/core/core.hpp>
#include "cuda_utils.h"
#include "NvInfer.h"


namespace vortex
{
    struct Rect
    {
        uint32_t x;
        uint32_t y;
        uint32_t width;
        uint32_t height;
    };

    std::ostream& operator<<(std::ostream& os, const Rect& rect);

    enum class Device
    {
        CPU,
        GPU,
    };

    /*
     * RGB Image in CPU/GPU
     */ 
    struct Image
    {
        uint8_t* dataCpu;
        uint8_t* dataGpu;
        uint8_t* data;

        uint32_t width;
        uint32_t height;
        uint32_t bytesPerLine;
        uint32_t channels = 3; // BGR image by default.
        // Device device;

        Image() : dataCpu(nullptr), dataGpu(nullptr), width(0), height(0), bytesPerLine(0), channels(0) {}
        
        Image(cv::Mat& image, bool copy = false)
        {
            cv::Mat img;
            if (!image.isContinuous())
                img = image.clone();
            else
                img = image;
            
            alloc(image.cols, image.rows, image.channels(), image.data);
        }

        Image(uint32_t width, uint32_t height, uint32_t channels)
        {
            alloc(width, height, channels);
        }

        ~Image() {}

        void clear()  // do not own memory.
        {
            delete dataCpu;
            dataCpu = nullptr;
            checkRuntime(cudaFree(dataGpu));
            dataGpu = nullptr;
            data = nullptr;
        }

        cv::Mat toCvMat() const
        {
            // cpu contains valid data
            return cv::Mat(height, width, CV_8UC3, data);
        }

        void alloc(uint32_t w, uint32_t h, uint32_t c, uint8_t* source = nullptr)
        {
            width = w;
            height = h;
            channels = c;
            bytesPerLine = width * channels;

            uint32_t data_size = w * h * c;
            dataCpu = new uint8_t[data_size];
            if (source != nullptr)
            {
                memcpy(dataCpu, source, data_size * sizeof(uint8_t));
            }
            else
            {
                memset(dataCpu, 0, data_size * sizeof(uint8_t));
            }

            checkRuntime(cudaMalloc(&dataGpu, data_size * sizeof(uint8_t)));
            checkRuntime(cudaMemcpy(dataGpu, dataCpu, data_size, cudaMemcpyHostToDevice));
            data = dataCpu;
        }

        void toCpu() { data = dataCpu; }
        void toGpu() { data = dataGpu; } 

        void syncCpu() 
        {
            if (dataGpu == nullptr)
                return;
            uint32_t data_size = width * height * channels;
            checkRuntime(cudaMemcpy(dataCpu, dataGpu, data_size, cudaMemcpyDeviceToHost));
            data = dataCpu;
        }

        void syncGpu() 
        {
            if (dataCpu == nullptr)
                return;
            uint32_t data_size = width * height * channels;
            checkRuntime(cudaMemcpy(dataGpu, dataCpu, data_size, cudaMemcpyHostToDevice));
            data = dataGpu;
        }

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
    
    std::map<std::string, nvinfer1::Weights> loadWeights(const std::string& filpath);
}
