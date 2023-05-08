#include "simple_infer_engine.h"
#include <fstream>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include "vortex/core/cuda_utils.h"
#include "vortex/utils/fileops.h"


namespace vortex
{
    SimpleInferEngine::~SimpleInferEngine()
    {
        if (m_Context != nullptr)
            m_Context->destroy();
        if (m_Engine != nullptr)
            m_Engine->destroy();
        if (m_Runtime != nullptr)
            m_Runtime->destroy();
    }

    bool SimpleInferEngine::LoadEngine(const std::string& engine_path, 
        const BlobInfo& input_info, const BlobInfo& output_info)
    {
        // load data from file
        std::vector<unsigned char> engine_data;
        bool ret = loadBinaryContent(engine_path, engine_data);
        if (!ret)
            return false;
        // create engine
        m_Runtime = nvinfer1::createInferRuntime(m_Logger);
        if (m_Runtime == nullptr)
            return false;
        m_Engine = m_Runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
        if (m_Engine == nullptr)
            return false;
        m_Context = m_Engine->createExecutionContext();
        if (m_Context == nullptr)
            return false;
        
        // create cuda stream
        checkRuntime(cudaStreamCreate(&m_Stream));

        m_InputInfo = input_info;
        m_OutputInfo = output_info;
        m_InputBlob = std::make_shared<BlobF>(input_info);
        m_OutputBlob = std::make_shared<BlobF>(output_info);

        return true;
    }

    void SimpleInferEngine::Infer(cv::Mat& image, std::vector<float>& output)
    {
        this->Preprocess(image);

        // set buffer data
        void* buffers[2] = { nullptr };
        const int inputIndex = m_Engine->getBindingIndex(m_InputBlob->name.c_str());
        const int outputIndex = m_Engine->getBindingIndex(m_OutputBlob->name.c_str());
        buffers[inputIndex] = m_InputBlob->dataGpu;
        buffers[outputIndex] = m_OutputBlob->dataGpu;

        m_InputBlob->ToGpuAsync(m_Stream);
        m_Context->enqueue(1, buffers, m_Stream, nullptr);
        output.resize(m_OutputBlob->count);
        m_OutputBlob->ToCpuAsync(m_Stream, output.data());
        cudaStreamSynchronize(m_Stream);
    }

    void SimpleInferEngine::Preprocess(cv::Mat& image)
    {
        // default preprocessing
        // resize, bgr2rgb, normalize
        cv::Mat temp;
        uint32_t width = m_InputInfo.width;
        uint32_t height = m_InputInfo.height;
        uint32_t channels = m_InputInfo.channels;
        cv::resize(image, temp, cv::Size(width, height));
        uint32_t image_area = width * height;
        uint32_t numel = image_area * channels;
        
        float* input_buffer = m_InputBlob->dataCpu;
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
}
