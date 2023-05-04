#pragma once
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include "vortex/core/logger.h"


namespace vortex
{
    class AlexNet
    {
    private:
        Logger m_Logger;
        // trt related
        nvinfer1::ICudaENgine* m_Engine;
        nvinfer1::IExecutionContext* m_Context;
        nvinfer1::IRuntime* m_Runtime;
        cudaStream_t m_Stream;  // cuda stream for synchronization.

        uint32_t m_InputWidth;
        uint32_t m_InputHeight;
        uint32_t m_InputChannels;
        uint32_t m_OutputSize;
        
        std::string m_InputBlobName;
        std::string m_OutputBlobName;
        std::vector<float> m_InputBuffer;
        float* m_InputBufferDevice;
        float* m_OutputBufferDevice;

    public:
        AlexNet(const std::string& weightsPath);
        ~AlextNet();

        void Infer(cv::Mat& image, std::vector<float>& output);
        void Preprocess(cv::Mat& image);
    };

}
