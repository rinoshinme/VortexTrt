#include "arcface.h"
#include <fstream>
#include <iostream>
#include "vortex/core/cuda_utils.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace vortex
{
    Arcface::Arcface(const std::string& model_path)
    {
        // load data from file
        std::vector<unsigned char> engine_data;
        LoadFile(model_path, engine_data);
        // assert(ret);

        // create engine
        m_Runtime = nvinfer1::createInferRuntime(m_Logger);
        assert(m_Runtime != nullptr);
        m_Engine = m_Runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
        assert(m_Engine != nullptr);
        m_Context = m_Engine->createExecutionContext();
        assert(m_Context != nullptr);

        m_InputBlobName = "input";
        m_OutputBlobName = "output";

        // allocate buffer for batchsize=1
        // only support batchsize = 1 for now.
        m_InputWidth = 112; 
        m_InputHeight = 112;
        m_InputChannels = 3;
        uint32_t input_numel = m_InputWidth * m_InputHeight * m_InputChannels;
        m_OutputSize = 512;
        m_InputBuffer.resize(input_numel);
        
        checkRuntime(cudaMalloc(&m_InputBufferDevice, input_numel * sizeof(float)));
        checkRuntime(cudaMalloc(&m_OutputBufferDevice, m_OutputSize * sizeof(float)));

        // create cuda stream
        checkRuntime(cudaStreamCreate(&m_Stream));
    }

    Arcface::~Arcface()
    {
        // destroy engine, execution context and cuda stream
        m_Context->destroy();
        m_Engine->destroy();
        m_Runtime->destroy();
        cudaStreamDestroy(m_Stream);

        // destroy buffers
        checkRuntime(cudaFree(m_InputBufferDevice));
        checkRuntime(cudaFree(m_OutputBufferDevice));
    }

    void Arcface::Preprocess(cv::Mat& image)
    {
        // preprocess
        cv::Mat temp;
        cv::resize(image, temp, cv::Size(m_InputWidth, m_InputWidth));

        uint32_t image_area = m_InputWidth * m_InputHeight;
        uint32_t input_numel = image_area * m_InputChannels;

        float* input_buffer = m_InputBuffer.data();

        float* pBlue = input_buffer;
        float* pGreen = input_buffer + image_area;
        float* pRed = input_buffer + image_area * 2;
        unsigned char* pImage = temp.data;
        for (uint32_t i = 0; i < image_area; ++i)
        {
            pRed[i] = (pImage[3 * i + 0] / 255.0 - 0.5) / 0.5;
            pGreen[i] = (pImage[3 * i + 1] / 255.0 - 0.5) / 0.5;
            pBlue[i] = (pImage[3 * i + 2] / 255.0 - 0.5) / 0.5;
        }
    }

    void Arcface::Infer(cv::Mat& image, std::vector<float>& output)
    {
        std::cout << "Start inference\n";
        // preprocess image data into input buffer
        Preprocess(image);

        // infer
        void* buffers[2] = { nullptr }; // for input/output buffer on gpu
        const int inputIndex = m_Engine->getBindingIndex("input");
        const int outputIndex = m_Engine->getBindingIndex("output");
        
        buffers[inputIndex] = m_InputBufferDevice;
        buffers[outputIndex] = m_OutputBufferDevice;

        uint32_t input_numel = m_InputWidth * m_InputHeight * m_InputChannels;
        checkRuntime(cudaMemcpyAsync(buffers[inputIndex], m_InputBuffer.data(), input_numel * sizeof(float), cudaMemcpyHostToDevice, m_Stream));
        m_Context->enqueue(1, buffers, m_Stream, nullptr);
        
        output.resize(m_OutputSize);

        checkRuntime(cudaMemcpyAsync(output.data(), buffers[outputIndex], m_OutputSize * sizeof(float), cudaMemcpyDeviceToHost, m_Stream));
        cudaStreamSynchronize(m_Stream);
    }

    bool Arcface::LoadFile(const std::string& file_path, std::vector<unsigned char>& data)
    {
        std::ifstream file(file_path, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size_t file_size = file.tellg();
            file.seekg(0, file.beg);
            data.resize(file_size);
            char* ptr = reinterpret_cast<char*>(data.data());
            file.read(ptr, file_size);
            file.close();
            return true;
        }
        return false;
    }
    
}
