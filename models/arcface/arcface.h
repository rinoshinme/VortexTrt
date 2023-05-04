#pragma once

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>

#include "vortex/core/logger.h"


namespace vortex
{
    class Arcface
    {
    private:
        Logger m_Logger;
        nvinfer1::ICudaEngine* m_Engine;
        nvinfer1::IExecutionContext* m_Context;
        nvinfer1::IRuntime* m_Runtime;

        std::string m_InputBlobName;
        std::string m_OutputBlobName;

        uint32_t m_InputWidth;
        uint32_t m_InputHeight;
        uint32_t m_InputChannels;
        uint32_t m_OutputSize;
        std::vector<float> m_InputBuffer;
        
        float* m_InputBufferDevice;
        float* m_OutputBufferDevice;

        cudaStream_t m_Stream;  // cuda stream for synchronization.

    public:
        Arcface(const std::string& model_path);
        ~Arcface();

        void Infer(cv::Mat& image, std::vector<float>& output);
        void Preprocess(cv::Mat& image);

    private:
        bool LoadFile(const std::string& file_path, std::vector<unsigned char>& data);
    };
}
