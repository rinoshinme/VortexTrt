#include "unet.h"
#include <opencv2/imgproc/imgproc.hpp>


namespace vortex
{
    Unet::Unet(const std::string& model_path)
    {
        m_InputBuffer = std::make_shared<Tensor4Df>(1, 3, 512, 512);
        uint32_t buffer_size = m_InputBuffer->Size();
        // equal input/output size, make sure that the 
        m_InputBufferDevice = std::make_shared<DeviceMemory>(buffer_size);
        m_OutputBufferDevice = std::make_shared<DeviceMemory>(buffer_size);
        LoadEngine(model_path);
        m_MaskThreshold = 0.5;
    }

    void Unet::Infer(cv::Mat& image, std::vector<float>& output)
    {
        Preprocess(image);

        // map buffer indices
        void* buffers[2] = { nullptr };
        const int inputIndex = m_Engine->getBindingIndex("input");
        const int outputIndex = m_Engine->getBindingIndex("output");
        buffers[inputIndex] = m_InputBufferDevice->Ptr();
        buffers[outputIndex] = m_OutputBufferDevice->Ptr();

        // copy data from host to device
        uint32_t data_size = m_InputBuffer->Size();
        checkRuntime(cudaMemcpyAsync(buffers[inputIndex], m_InputBuffer->Ptr(), 
            data_size * sizeof(float), cudaMemcpyHostToDevice, m_Stream));
        m_Context->enqueue(1, buffers, m_Stream, nullptr);

        // copy results from device to host
        output.resize(data_size);
        checkRuntime(cudaMemcpyAsync(output.data(), buffers[outputIndex], 
            data_size * sizeof(float), cudaMemcpyDeviceToHost, m_Stream);
    }

    void Unet::Preprocess(cv::Mat& image)
    {
        uint32_t width = m_InputBuffer->ImageWidth();
        uint32_t height = m_InputBuffer->ImageHeight();
        cv::Mat temp;
        cv::resize(image, temp, cv::Size(width, height));
        // copy data to input buffer
        uint32_t image_area = m_InputBuffer->ImageArea();
        float* input_buffer = m_InputBuffer->Ptr();
        float* pBlue = input_buffer;
        float* pGreen = input_buffer + image_area;
        float* pRed = input_buffer + image_area * 2;
        unsigned char* pImage = temp.data;

        for (uint32_t i = 0; i < image_area; ++i)
        {
            pRed[i] = pImage[3 * i + 0] / 255.0;
            pGreen[i] = pImage[3 * i + 1] / 255.0;
            pBlue[i] = pImage[3 * i + 2] / 255.0;
        }
    }
}
